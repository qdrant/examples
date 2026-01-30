package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/qdrant/go-client/qdrant"
)

const (
	collectionName = "products"
	version        = "1.0.0"
)

var (
	qdrantClient *qdrant.Client
	openaiAPIKey string
)

func main() {
	qdrantHost := os.Getenv("QDRANT_HOST")
	qdrantAPIKey := os.Getenv("QDRANT_API_KEY")
	openaiAPIKey = os.Getenv("OPENAI_API_KEY")

	if qdrantHost == "" || qdrantAPIKey == "" || openaiAPIKey == "" {
		log.Fatal("QDRANT_HOST, QDRANT_API_KEY, and OPENAI_API_KEY environment variables must be set")
	}

	// Connect to Qdrant
	var err error
	qdrantClient, err = qdrant.NewClient(&qdrant.Config{
		Host:   qdrantHost,
		APIKey: qdrantAPIKey,
		Port:   6334,
		UseTLS: true,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Qdrant: %v", err)
	}
	defer qdrantClient.Close()

	// Setup Gin router
	r := gin.Default()

	r.GET("/health", healthHandler)
	r.POST("/search", searchHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"version": version,
	})
}

type SearchRequest struct {
	Query string `json:"query" binding:"required"`
	TopK  int    `json:"top_k"`
}

type ProductPayload struct {
	ArticleID       string `json:"article_id"`
	ProdName        string `json:"prod_name"`
	ProductTypeName string `json:"product_type_name"`
	ColourGroupName string `json:"colour_group_name"`
	DetailDesc      string `json:"detail_desc"`
	ImageURL        string `json:"image_url"`
}

type SearchPoint struct {
	ID      uint64         `json:"id"`
	Payload ProductPayload `json:"payload"`
	Score   float32        `json:"score"`
}

func searchHandler(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query field is required"})
		return
	}

	topK := req.TopK
	if topK <= 0 {
		topK = 10
	}
	if topK > 100 {
		topK = 100
	}

	ctx := context.Background()

	// Query Qdrant using document-based vector (cloud inference)
	points, err := qdrantClient.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query: qdrant.NewQueryNearest(
			qdrant.NewVectorInputDocument(&qdrant.Document{
				Text:  req.Query,
				Model: "openai/text-embedding-3-small",
				Options: map[string]*qdrant.Value{
					"openai-api-key": qdrant.NewValueString(openaiAPIKey),
				},
			}),
		),
		Limit:       qdrant.PtrOf(uint64(topK)),
		WithPayload: qdrant.NewWithPayload(true),
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Transform results
	searchPoints := make([]SearchPoint, 0, len(points))
	for _, point := range points {
		payload := point.Payload
		searchPoints = append(searchPoints, SearchPoint{
			ID: point.Id.GetNum(),
			Payload: ProductPayload{
				ArticleID:       getStringPayload(payload, "article_id"),
				ProdName:        getStringPayload(payload, "prod_name"),
				ProductTypeName: getStringPayload(payload, "product_type_name"),
				ColourGroupName: getStringPayload(payload, "colour_group_name"),
				DetailDesc:      getStringPayload(payload, "detail_desc"),
				ImageURL:        getStringPayload(payload, "image_url"),
			},
			Score: point.Score,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"query": req.Query,
		"results": gin.H{
			"points": searchPoints,
		},
	})
}

func getStringPayload(payload map[string]*qdrant.Value, key string) string {
	if v, ok := payload[key]; ok {
		return v.GetStringValue()
	}
	return ""
}