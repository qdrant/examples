### Movie recommendation system with Qdrant space vectors

In the pursuit of creating a movie recommendation system, we'll leverage the MovieLens dataset alongside Qdrant's capabilities. The key to our system lies in the method known as **collaborative filtering**. The premise is straightforward yet powerful: users with similar movie preferences are likely to appreciate the same films. Through this angle, the system aims to match users with similar tastes and recommend movies liked by these akin users that one has yet to watch.

Here's an outline of how we'll implement this concept:

1. **User Ratings as Vectors**: The movie ratings by each user are transformed into vectors. These vectors exist in a sparse, high-dimensional space - a necessary approach to encapsulate the wide variance in user preferences and movie characteristics.

2. **Indexing with Qdrant**: To efficiently manage and query these vectors, we utilize Qdrant, a potent tool for indexing high-dimensional space vectors. It facilitates the quick retrieval of similar vectors - in our case, users with similar tastes.

3. **Finding Similar Users**: With the vectors indexed, we exploit Qdrant's search capabilities to identify users whose ratings closely match our own. This step is crucial for uncovering user clusters with common preferences.

4. **Movie Recommendations**: Based on the similarities found, we investigate the movies favored by these like-minded individuals that the original user hasn't seen. This process forms the core of our recommendation engine, promising to suggest movies that are likely to resonate.

Before delving into the technical execution, it's necessary to set up the environment with the required tools - notably, the `qdrant-client` for interfacing with Qdrant and `pandas` for data manipulation. These libraries are installed using the package manager `pip`, which fetches and installs software packages from Python's index and other repositories. This step ensures that we have all the dependencies ready for our recommendation system project to come to life.

```python
!pip install qdrant-client pandas
```

### Download and unzip the dataset

In this initial step, the objective is to prepare the local environment for analysis by retrieving the required dataset. This involves creating a directory named "data" to store the dataset. Using the `wget` command, the MovieLens 1M dataset, hosted at `https://files.grouplens.org/datasets/movielens/ml-1m.zip`, is downloaded. Following the download, the `unzip` command is employed to extract the contents of the `ml-1m.zip` file into the newly created "data" directory. This process ensures that the dataset is readily accessible for further operations, such as data exploration or model training. The MovieLens 1M dataset is commonly used in recommendation system projects for its rich collection of movie ratings.

```python
# Download and unzip the dataset

!mkdir -p data
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip -d data
```

---

### Introduction to Qdrant Client and Pandas

In this section, we're introduced to the foundational elements required to work with Qdrant, a vector search engine, and to manipulate data efficiently. The Python libraries involved are the `qdrant_client` and `pandas`.

- **QdrantClient** from `qdrant_client`: This is the primary interface to interact with the Qdrant service. It provides functionalities for connecting to a Qdrant server, managing collections (datasets), inserting vectors, and making search queries among others. Qdrant specializes in handling vector data, which is essential for many machine learning and data retrieval applications.

- **Pandas**: A widely used library in data science and analytics, `pandas` offers data structures and operations for manipulating numerical tables and time series. It's especially recognized for its `DataFrame` object, which allows for easy data manipulation, aggregation, and visualization.

This combination of libraries empowers users to manage vector data with Qdrant while utilizing pandas for data preprocessing, exploration, and analysis. Here, their imports signify the initiation of a workflow that leverages the strengths of both tools to handle vector search and data manipulation tasks efficiently.

```python
from qdrant_client import QdrantClient, models
import pandas as pd
```

These lines of code are typically the stepping stone in a project that aims to leverage Qdrant's vector search capabilities with the robust data manipulation features of pandas.

---

```python
from qdrant_client import QdrantClient, models
import pandas as pd
```

### Loading User Data

The process begins by importing user data from a CSV file, showcasing the practical approach of handling data within a Jupyter notebook environment. Here, a dataset called `users.dat` is loaded into a pandas DataFrame. This DataFrame is constructed with specific details about users, including their `user_id`, `gender`, `age`, `occupation`, and `zip` code. The data is separated by a double colon (`::`), necessitating the specification of the separator (`sep='::'`) during the loading process. The use of the `engine='python'` parameter is required to correctly process this non-standard delimiter.

This step is crucial as it forms the foundation of data exploration and manipulation by providing a structured and accessible view of the data within Python. The DataFrame `users` now contains the loaded data, although the output was not explicitly shown in the notebook, indicating the successful loading of data without an immediate verification step in the provided snippet.

```python
users = pd.read_csv(
    './data/ml-1m/users.dat',
    sep='::',
    names=[
        'user_id',
        'gender',
        'age',
        'occupation',
        'zip'
    ],
    engine='python'
)
users
```

Outputs: ['']

### Loading Movie Data

In this step, we utilize Pandas, a popular Python library for data manipulation and analysis, to load a dataset containing movies. The dataset is read from a `movies.dat` file which is located in the `./data/ml-1m/` directory. The file is structured in a format where each field is separated by `::`, a non-standard delimiter requiring specification during the loading process.

The columns included in the dataset are:
- `movie_id`: A unique identifier for each movie.
- `title`: The title of the movie.
- `genres`: Genres that the movie belongs to, potentially multiple, separated by a certain character.

The data is loaded into a DataFrame named `movies`, a table-like data structure provided by Pandas. The `engine='python'` argument is specified due to the use of the non-standard delimiter, as the default C engine does not support it. The file's encoding is set to `'latin-1'` to correctly handle special characters within the movie titles or genres.

This step is essential for preparing the dataset for further analysis or processing, such as filtering based on genres, searching for particular titles, or preparing the data for machine learning tasks.

```python
movies = pd.read_csv(
    './data/ml-1m/movies.dat',
    sep='::',
    names=[
        'movie_id',
        'title',
        'genres'
    ],
    engine='python',
    encoding='latin-1'
)
movies
```

Outputs: ['']

### Loading the Ratings Dataset

Understanding user preferences and behaviors is critical in many domains, especially in recommendation systems. One of the fundamental datasets in this domain is the MovieLens 1M dataset, which contains 1 million ratings from thousands of users on various movies. To analyze this dataset, the first step is to load it into a structure that allows for efficient manipulation and analysis.

The chosen method to accomplish this is by using the Pandas library in Python, known for its powerful data manipulation capabilities. The code snippet provided reads the `ratings.dat` file from the MovieLens 1M dataset. The data is structured with each row containing a user's rating for a movie and the time at which the rating was given. This is done using the `pd.read_csv()` function, indicating the column separator as `::` and naming the columns as 'user_id', 'movie_id', 'rating', and 'timestamp'. Due to the unconventional separator, the `engine` parameter is set to 'python' to handle the parsing correctly.

By loading the dataset in this manner, one sets up a foundational step towards deeper analysis, such as understanding user preferences, identifying popular movies, or even building personalized recommendation systems. This approach emphasizes the importance of efficient data loading techniques as a precursor to complex data analysis and machine learning tasks.

```python
ratings = pd.read_csv(
    './data/ml-1m/ratings.dat',
    sep='::',
    names=[
        'user_id',
        'movie_id',
        'rating',
        'timestamp'
    ],
    engine='python'
)
```

### Normalize ratings

When handling user ratings in recommendation systems, it's common to encounter skewed or unstandardized data. Normalizing these ratings is a crucial step to ensure that the model accurately interprets user preferences, especially when dealing with sparse data sets where most values might be missing or zeros. This method transforms the ratings to have a mean of 0 and a standard deviation of 1. By doing so, not only are the ratings standardized across the entire dataset, but it also allows the model to effectively incorporate movies with negative feedback (i.e., those that a user didn't like). 

This normalization is achieved through the following line of Python code:

```python
ratings.rating = (ratings.rating - ratings.rating.mean()) / ratings.rating.std()
```

Here, each rating is adjusted by subtracting the mean rating of the entire dataset and then dividing by the standard deviation. This technique effectively centers the data around 0 and scales it in terms of standard deviation. Such a transformation is particularly useful in sparse vector scenarios typical in recommendation systems, as it enhances model sensitivity to both positive and negative user preferences.

```python
# Normalize ratings

# Sparse vectors can use advantage of negative values, so we can normalize ratings to have mean 0 and std 1
# In this scenario we can take into account movies that we don't like

ratings.rating = (ratings.rating - ratings.rating.mean()) / ratings.rating.std()
```

### Convert ratings to sparse vectors

In this section, the code is focused on converting user movie ratings into sparse vectors. This is a necessary preprocessing step for many machine learning models, especially when dealing with large datasets as it helps in reducing memory usage and computation time. Sparse vectors only store non-zero elements, which is efficient given that not every user has rated every movie in a typical movie rating dataset.

To accomplish this, the code employs a `defaultdict` from Python's collections module to create a dictionary (`user_sparse_vectors`) where each key represents a unique user ID. The values are dictionaries themselves, containing two lists: `values` for storing the ratings and `indices` for storing the corresponding movie IDs. This structure is chosen because a sparse vector can be effectively represented by just keeping track of the indices where there are non-zero elements (in this case, the movie IDs the user has rated) and the values at those indices (the user's ratings for those movies).

The loop iterates over each row in the `ratings` data (presumably a DataFrame object holding user IDs, movie IDs, and ratings), appending the rating to the `values` list and the movie ID to the `indices` list for the corresponding user ID. This process effectively transforms the dense representation of the ratings into a sparse format, where the majority of the matrix, which would have been zeroes, is not explicitly stored.

This transformation is particularly useful in scenarios where the dataset is vast and sparse, such as recommendation systems in which there are thousands of users and items but only a small fraction of all possible user-item pairs have interactions (ratings, views, purchases, etc.). Sparse representations help in significantly reducing the computational load and memory requirements for storing and processing this data.

```python
# Convert ratings to sparse vectors

from collections import defaultdict

user_sparse_vectors = defaultdict(lambda: {
    "values": [],
    "indices": []
})

for row in ratings.itertuples():
    user_sparse_vectors[row.user_id]["values"].append(row.rating)
    user_sparse_vectors[row.user_id]["indices"].append(row.movie_id)
```

### For this small dataset we can use in-memory Qdrant

In this section, the focus is on initializing a Qdrant client for handling a small dataset. This approach is demonstrated through the initialization of the client with `QdrantClient(":memory:")`. This implements Numpy-based in-memory storage, which is suitable for small datasets and is particularly useful for prototyping and testing purposes. 

However, it's also mentioned that for production environments or applications dealing with larger datasets, a server-based version is recommended. This would involve setting up a Qdrant server and connecting to it via its URL, like `QdrantClient("http://localhost:6333")`. This setup allows for handling larger datasets, offering more robust data management, scalability, and possibly distributed processing capabilities. 

The choice between in-memory and server-based configurations is crucial for optimizing performance and resource utilization based on the project's specific needs.

```python
# For this small dataset we can use in-memory Qdrant
# But for production we recommend to use server-based version

qdrant = QdrantClient(":memory:") # or QdrantClient("http://localhost:6333")
```

### Create Collection with Configured Sparse Vectors

In this section, the focus is on initializing a collection within Qdrant — a vector search engine that enables efficient storage and search of high-dimensional vectors. The collection, named "movielens", is configured to support sparse vectors. Unlike dense vectors, sparse vectors are characterized by high dimensionality with most of the elements being zero. These vectors are particularly useful in scenarios where data is inherently sparse, such as user-item interaction matrices in recommendation systems.

The code demonstrates how to create a collection by invoking the `create_collection` method on the Qdrant client. The `vectors_config` is left empty, indicating no configuration for dense vectors, while `sparse_vectors_config` is populated to include a "ratings" field, utilizing the `SparseVectorParams` function. By doing this, the collection is specifically tailored to handle sparse vectors under the "ratings" field without needing to predefine the dimensions. This flexibility is significant because it allows the collection to automatically extract and accommodate the dimensionality of the incoming sparse vector data.

The output `['']` indicates that the collection was created successfully without any errors. This step is foundational in setting up a structure within Qdrant that can efficiently handle sparse vector data, enabling operations like similarity search, which are central to recommendation systems and other applications dealing with sparse high-dimensional data.

```python
# Create collection with configured sparse vectors
# Sparse vectors don't require to specify dimension, because it's extracted from the data automatically

qdrant.create_collection(
    "movielens",
    vectors_config={},
    sparse_vectors_config={
        "ratings": models.SparseVectorParams()
    }
)
```

Outputs: ['']

### Upload all user's votes as sparse vectors

In this section, the process of uploading user votes represented as sparse vectors to the Qdrant database is outlined. The given code snippet introduces a Python function, `data_generator`, designed to iteratively prepare data from a dataset of users for upload. 

The structure of each user's data is encapsulated by `models.PointStruct`, which includes a unique identifier (`id` corresponding to `user_id`), a vector that represents the user's ratings in a sparse format (`"ratings": user_sparse_vectors[user.user_id]`), and additional payload carrying the full user data (`user._asdict()`).

Following the data preparation, the `qdrant.upload_points` method is invoked, specifying the target collection ("movielens") and passing the generator function (`data_generator()`), which feeds the prepared data for upload. This approach is noted for its efficiency, as it performs a lazy upload, implying that data is processed and uploaded in an on-demand fashion without requiring all of it to be loaded into memory upfront.

This method is particularly useful when dealing with large datasets of user interactions, such as ratings or votes, by storing them in a structured and query-able format within Qdrant. By converting user votes into sparse vectors, the solution optimizes storage and speeds up the querying process, making it an effective tool for handling user-generated data and enhancing recommendation systems or user analysis workflows.

```python
# Upload all user's votes as sparse vectors

def data_generator():
    for user in users.itertuples():
        yield models.PointStruct(
            id=user.user_id,
            vector={
                "ratings": user_sparse_vectors[user.user_id]
            },
            payload=user._asdict()
        )

# This will do lazy upload of the data
qdrant.upload_points(
    "movielens",
    data_generator()
)
```

### Let's try to recommend something for ourselves

In this segment, the intention is to recommend movies based on a personal rating system. In this scenario, a rating of `1` signifies liking a movie, while `-1` represents dislike. A Python dictionary, `my_ratings`, enumerates several movies by their unique identifiers, alongside these like or dislike ratings. The movies cited range from science fiction and fantasy to classics and thrillers, including titles such as "The Matrix," "Star Trek," "Star Wars," "The Thing," "Toy Story," "Titanic," "Pulp Fiction," "Forrest Gump," "Lord of the Rings," "Indiana Jones," and "Die Hard."

A critical component of this process involves transforming these personal ratings into a format that's usable for modeling or recommending systems. This is achieved through the `to_vector` function, which constructs a sparse vector from the ratings. Sparse vectors are a data-efficient way to store information when a large proportion of the elements are zero, which is often the case in recommendation systems where a user has only interacted with a small subset of items.

The `inverse_ratings` dictionary, created through a dictionary comprehension that flips the sign of each rating, appears to be a preparatory step for a recommendation model. Inverting the ratings might be used in a scenario where the model needs to understand negative preferences explicitly, rather than merely the absence of a positive rating.

This example demonstrates an initial approach to handling personalized data in preparation for more complex recommendation engine tasks. It emphasizes not just the importance of capturing a user's preferences (likes and dislikes) but also the preliminary data transformation required to fit into a recommendation model's input format. This step is foundational for any recommendation system looking to provide tailored suggestions based on individual user ratings.

```python
# Let's try to recommend something for ourselves

#  1 - like
# -1 - dislike

# Search with 
# movies[movies.title.str.contains("Matrix", case=False)]

my_ratings = { 
    2571: 1,  # Matrix
    329: 1,   # Star Trek
    260: 1,   # Star Wars
    2288: -1, # The Thing
    1: 1,     # Toy Story
    1721: -1, # Titanic
    296: -1,  # Pulp Fiction
    356: 1,   # Forrest Gump
    2116: 1,  # Lord of the Rings
    1291: -1, # Indiana Jones
    1036: -1  # Die Hard
}

inverse_ratings = {k: -v for k, v in my_ratings.items()}

def to_vector(ratings):
    vector = models.SparseVector(
        values=[],
        indices=[]
    )
    for movie_id, rating in ratings.items():
        vector.values.append(rating)
        vector.indices.append(movie_id)
    return vector

```

### Find Users with Similar Taste

In this section, we are exploring how to identify users who share a similar taste in movies. To achieve this, we leverage the functionality provided by Qdrant, a vector search engine designed for efficient similarity search in large datasets. The focus is on utilizing user ratings to find matches, which suggests a collaborative filtering approach to recommend movies.

The code snippet demonstrates the use of Qdrant's `search` method to perform this task. Here, `qdrant.search` is called with several parameters specifying how to find these similar users:
- `"movielens"`: This appears to be the dataset or collection within Qdrant where user ratings or profiles are stored.
- `query_vector=models.NamedSparseVector(...)`: A `NamedSparseVector` object is being created and passed as the query vector. This object likely represents the current user's movie rating profile, with `name="ratings"` specifying the nature of this vector, and `vector=to_vector(my_ratings)` converting the user's ratings into the proper format for the query.
- `with_vectors=True`: This flag indicates that the results should include the matched users' vectors. These vectors are important for the subsequent steps, perhaps for recommending new movies based on the tastes of similar users.
- `limit=20`: This limits the search to the top 20 users with the most similar taste in movies.

Through this process, Qdrant enables the identification of users with similar movie preferences, utilizing their ratings as a vector for comparison. This methodology forms the basis for potential recommendation systems, where understanding shared interests among users can help in suggesting new and enjoyable content.

```python
# Find users with similar taste

results = qdrant.search(
    "movielens",
    query_vector=models.NamedSparseVector(
        name="ratings",
        vector=to_vector(my_ratings)
    ),
    with_vectors=True, # We will use those to find new movies
    limit=20
)

```

### Calculate how frequently each movie is found in similar users' ratings

This section of the blog post delves into a crucial aspect of recommendation systems: identifying movies that are frequently rated by users similar to a given user. The primary goal here is to leverage user similarity to recommend movies that a user might enjoy, based on their viewing habits and preferences.

To achieve this, a function `results_to_scores` is defined. This function plays a pivotal role in transforming the outcomes of user similarity searches into actionable insights—namely, scores that represent how frequently each movie is found in similar users' ratings. This is a fundamental step in collaborative filtering, where recommendations are made by considering the preferences of users who are deemed similar.

Here's how the function operates in detail:

1. **Initialization of a Movie Scores Dictionary**: A `defaultdict` is used to initialize a dictionary (`movie_scores`) that maps movie indices to their respective scores. The use of `defaultdict(lambda: 0)` ensures that movies not previously encountered are automatically assigned a score of 0, simplifying increment operations.

2. **Iterating Through Similar Users' Ratings**: The function iterates through each similar user's ratings, represented as vectors. Each vector consists of indices corresponding to movie IDs and values corresponding to the ratings these movies have received.

3. **Updating Scores While Ignoring Movies Rated by the Current User**: For each similar user, the function iterates through their rated movies. It then updates the `movie_scores` dictionary by adding the rating value to the score associated with each movie's index. Movies already rated by the querying user (`my_ratings`) are skipped to ensure that only new recommendations are considered.

This approach effectively aggregates the preferences of similar users, thereby identifying movies that are popular or highly regarded within the user's similarity cluster. It's an essential component of building personalized recommendation systems that cater to individual tastes by exploiting the collective intelligence of the user base.

```python
# Calculate how frequently each movie is found in similar users' ratings

def results_to_scores(results):
    movie_scores = defaultdict(lambda: 0)

    for user in results:
        user_scores = user.vector['ratings']
        for idx, rating in zip(user_scores.indices, user_scores.values):
            if idx in my_ratings:
                continue
            movie_scores[idx] += rating

    return movie_scores
```

### Sort movies by score and print top 5

This section demonstrates how to rank movies based on a scoring system and identify the top 5. By taking a dictionary or similar structure (`movie_scores`) that associates movies with their respective scores, the code efficiently sorts these movies in descending order of their scores. This means the movies with the highest scores come first. Here, a lambda function is used as the key for sorting, which specifies that the sorting should be based on the scores (the second element in each item of the `movie_scores` dictionary). 

After sorting, the code snippet prints out the titles and scores of the top 5 movies. The movies printed, along with their scores, provide insight into the effectiveness of the scoring system used. For instance, "Star Wars: Episode V - The Empire Strikes Back (1980)" comes out on top with a score of approximately 20.02, followed by other high-ranking movies such as "Star Wars: Episode VI - Return of the Jedi (1983)" and "Princess Bride, The (1987)." 

This output showcases not only the power of sorting and lambda functions in Python but also underlines the movies that are potentially the most relevant or of the highest quality according to the scoring system applied. Furthermore, this approach could be utilized in various applications, including recommendation systems, where determining the top items (in this case, movies) based on certain criteria is crucial.

```python
# Sort movies by score and print top 5

movie_scores = results_to_scores(results)
top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

for movie_id, score in top_movies[:5]:
    print(movies[movies.movie_id == movie_id].title.values[0], score)
```

Outputs: ['Star Wars: Episode V - The Empire Strikes Back (1980) 20.023877887283938\nStar Wars: Episode VI - Return of the Jedi (1983) 16.44318377549194\nPrincess Bride, The (1987) 15.84006760423755\nRaiders of the Lost Ark (1981) 14.94489407628955\nSixth Sense, The (1999) 14.570321651488953\n']

## Finding Users with Similar Taste Within Specific Age Groups

In an innovative example of personalized content recommendation, Qdrant can be used to find users with similar tastes in movies, with the added ability to filter these results by specific criteria such as age, gender, or occupation. This process not only tailors recommendations more accurately but also introduces a layer of demographic specificity that goes beyond the general matching of tastes.

The code example demonstrates how to query the Qdrant database for movie recommendations based on a user's individual movie ratings. The query incorporates a user's ratings into a vector format and applies an age filter to ensure recommendations are relevant to the user's age group. This is particularly useful for capturing the nuances of preference that can vary significantly across different age demographics.

The significance of including a filter based on the `age` field is highlighted by the result set, which presents movies that presumably resonate well with users in the specified age group. The output reveals a diverse array of popular films such as "The Princess Bride (1987)", "Star Wars: Episode V - The Empire Strikes Back (1980)", and "The Godfather (1972)", ranked by their relevance scores.

These scores, computed by Qdrant, reflect how closely the movies align with the user's tastes, filtered through the lens of age similarity. The results illustrate the tool's ability to discern and prioritize movies that not only match a user's general tastes but also conform to patterns and preferences typical of their age group. This approach enhances the personalization of recommendations, potentially improving user satisfaction by acknowledging the complex factors that influence movie preferences.

```python
# Find users with similar taste, but only within my age group
# We can also filter by other fields, like `gender`, `occupation`, etc.

results = qdrant.search(
    "movielens",
    query_vector=models.NamedSparseVector(
        name="ratings",
        vector=to_vector(my_ratings)
    ),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="age",
                match=models.MatchValue(value=25)
            )
        ]
    ),
    with_vectors=True,
    limit=20
)

movie_scores = results_to_scores(results)
top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

for movie_id, score in top_movies[:5]:
    print(movies[movies.movie_id == movie_id].title.values[0], score)
```

Outputs: ['Princess Bride, The (1987) 16.214640029038147\nStar Wars: Episode V - The Empire Strikes Back (1980) 14.652836719595939\nBlade Runner (1982) 13.52911944519415\nUsual Suspects, The (1995) 13.446604377087162\nGodfather, The (1972) 13.300575698740357\n']

