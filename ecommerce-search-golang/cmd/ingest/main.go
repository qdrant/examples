package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"

	_ "github.com/duckdb/duckdb-go/v2"
	"github.com/qdrant/go-client/qdrant"
	"github.com/schollz/progressbar/v3"
)

const (
	collectionName    = "products"
	batchSize         = 100 // Larger batches = fewer API calls
	defaultNumWorkers = 4   // Parallel upsert workers
	// DuckDB can read parquet directly from Hugging Face
	datasetURL = "https://huggingface.co/api/datasets/Qdrant/hm_ecommerce_products/parquet/default/train/0.parquet"
)

func main() {
	qdrantHost := os.Getenv("QDRANT_HOST")
	qdrantAPIKey := os.Getenv("QDRANT_API_KEY")
	openaiAPIKey := os.Getenv("OPENAI_API_KEY")

	if qdrantHost == "" || qdrantAPIKey == "" || openaiAPIKey == "" {
		log.Fatal("QDRANT_HOST, QDRANT_API_KEY, and OPENAI_API_KEY environment variables must be set")
	}

	numWorkers := defaultNumWorkers
	if nw := os.Getenv("NUM_WORKERS"); nw != "" {
		if n, err := strconv.Atoi(nw); err == nil && n > 0 {
			numWorkers = n
		}
	}
	log.Printf("Using %d workers", numWorkers)

	ctx := context.Background()

	// Connect to Qdrant
	log.Println("Connecting to Qdrant...")
	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host:   qdrantHost,
		APIKey: qdrantAPIKey,
		Port:   6334,
		UseTLS: true,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Qdrant: %v", err)
	}
	defer qdrantClient.Close()

	// Create collection
	if err := createCollection(ctx, qdrantClient); err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Open DuckDB (in-memory)
	log.Println("Opening DuckDB...")
	db, err := sql.Open("duckdb", "")
	if err != nil {
		log.Fatalf("Failed to open DuckDB: %v", err)
	}
	defer db.Close()

	// Install and load httpfs extension for HTTP parquet access
	log.Println("Installing and loading httpfs extension...")
	if _, err := db.Exec("INSTALL httpfs; LOAD httpfs;"); err != nil {
		log.Fatalf("Failed to load httpfs extension: %v", err)
	}

	// Stream products from Hugging Face and upsert to Qdrant
	if err := streamAndUpsert(ctx, db, qdrantClient, openaiAPIKey, numWorkers); err != nil {
		log.Fatalf("Failed to ingest data: %v", err)
	}

	fmt.Println("\nData ingestion complete!")
}

func createCollection(ctx context.Context, client *qdrant.Client) error {
	exists, err := client.CollectionExists(ctx, collectionName)
	if err != nil {
		return err
	}

	if exists {
		log.Printf("Collection '%s' already exists, skipping creation", collectionName)
		return nil
	}

	err = client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     1536, // text-embedding-3-small dimension
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil {
		return err
	}

	log.Printf("Created collection: %s", collectionName)
	return nil
}

func streamAndUpsert(ctx context.Context, db *sql.DB, qdrantClient *qdrant.Client, openaiAPIKey string, numWorkers int) error {
	// Get total count first for progress bar
	log.Println("Getting total row count...")
	var totalRows int
	countQuery := `SELECT COUNT(*) FROM read_parquet('` + datasetURL + `')`
	if err := db.QueryRow(countQuery).Scan(&totalRows); err != nil {
		return fmt.Errorf("failed to get row count: %w", err)
	}
	log.Printf("Total products to ingest: %d", totalRows)

	// Query parquet directly from Hugging Face URL
	query := `
		SELECT
			article_id,
			prod_name,
			product_type_name,
			colour_group_name,
			detail_desc,
			image_url
		FROM read_parquet('` + datasetURL + `')
	`

	rows, err := db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	// Create progress bar
	bar := progressbar.NewOptions(totalRows,
		progressbar.OptionSetDescription("Ingesting products"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "=",
			SaucerHead:    ">",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetWidth(40),
	)

	// Channel for batches to upsert
	batchChan := make(chan []*qdrant.PointStruct, numWorkers*2)
	errChan := make(chan error, 1)
	var wg sync.WaitGroup

	// Start worker goroutines
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range batchChan {
				if _, err := qdrantClient.GetPointsClient().Upsert(ctx, &qdrant.UpsertPoints{
					CollectionName: collectionName,
					Points:         batch,
				}); err != nil {
					select {
					case errChan <- err:
					default:
					}
					return
				}
				bar.Add(len(batch))
			}
		}()
	}

	var points []*qdrant.PointStruct
	var totalCount int

	for rows.Next() {
		// Check for worker errors
		select {
		case err := <-errChan:
			close(batchChan)
			return err
		default:
		}

		var (
			articleID       string
			prodName        string
			productTypeName string
			colourGroupName string
			detailDesc      string
			imageURL        string
		)

		err := rows.Scan(
			&articleID,
			&prodName,
			&productTypeName,
			&colourGroupName,
			&detailDesc,
			&imageURL,
		)
		if err != nil {
			continue
		}

		// Create text for embedding: combine name and description
		text := fmt.Sprintf("%s - %s - %s", prodName, productTypeName, detailDesc)

		totalCount++
		point := &qdrant.PointStruct{
			Id: qdrant.NewIDNum(uint64(totalCount)),
			Vectors: qdrant.NewVectorsDocument(&qdrant.Document{
				Text:  text,
				Model: "openai/text-embedding-3-small",
				Options: map[string]*qdrant.Value{
					"openai-api-key": qdrant.NewValueString(openaiAPIKey),
				},
			}),
			Payload: qdrant.NewValueMap(map[string]any{
				"article_id":        articleID,
				"prod_name":         prodName,
				"product_type_name": productTypeName,
				"colour_group_name": colourGroupName,
				"detail_desc":       detailDesc,
				"image_url":         imageURL,
			}),
		}
		points = append(points, point)

		// Send batch to workers
		if len(points) >= batchSize {
			batchChan <- points
			points = make([]*qdrant.PointStruct, 0, batchSize)
		}
	}

	// Send remaining points
	if len(points) > 0 {
		batchChan <- points
	}

	// Close channel and wait for workers
	close(batchChan)
	wg.Wait()

	// Check for any final errors
	select {
	case err := <-errChan:
		return err
	default:
	}

	if err := rows.Err(); err != nil {
		return err
	}

	return nil
}