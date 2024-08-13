# 🎥 Movie Recommendation System

![img](https://github.com/IrfanKpm/mini-ml-projects/blob/main/images/movie.png)


Welcome to the Movie Recommendation System project! In this project, we build a content-based recommendation system using various machine learning techniques to suggest movies based on user preferences. This README will walk you through the theories and algorithms that power the recommendation system, providing a comprehensive understanding of how it works.

## 📚 Theories and Concepts Behind the Project

### 1. **Content-Based Filtering**
Content-based filtering is a method of recommending items to users based on the attributes of the items themselves. In this case, the system uses features such as genres, keywords, tagline, cast, and director of movies to find similarities between them. The fundamental idea is that if a user likes a particular movie, they will likely enjoy other movies with similar attributes.

### 2. **Textual Data Representation**
To compare movies based on their features, we need to represent textual data (like genres or keywords) in a numerical format that the machine learning algorithms can process. This is done using a technique called **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 3. **TF-IDF Vectorization**
TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). The TF-IDF score increases with the number of times a word appears in a document, but is offset by the frequency of the word in the corpus. This helps in identifying the most significant words in the context of the entire dataset.

**TF-IDF Formula**:

TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
- **TF(t, d)**: Term Frequency of term `t` in document `d`.
- **IDF(t)**: Inverse Document Frequency of term `t` in the corpus.

By vectorizing our movie features using TF-IDF, we can convert the textual data into a matrix of numerical values where each element represents the TF-IDF score of a word in a movie's feature set.

### 4. **Cosine Similarity**
Once the movies are represented as vectors, we need a way to measure the similarity between these vectors. This is where **Cosine Similarity** comes into play.

**Cosine Similarity** measures the cosine of the angle between two vectors. It gives a value between -1 and 1, where:
- **1** indicates that the vectors are identical (i.e., the movies are very similar).
- **0** indicates that the vectors are orthogonal (i.e., no similarity).
- **-1** indicates that the vectors are diametrically opposite (i.e., they are completely dissimilar).

The formula for cosine similarity is:

Cosine Similarity(A, B) = (A · B) / (||A|| ||B||)

Where:
- **A** and **B** are the feature vectors of two movies.
- **A · B** is the dot product of the vectors.
- **||A||** and **||B||** are the magnitudes of the vectors.

Cosine Similarity is particularly useful in text analysis because it is independent of the magnitude of the vectors, focusing purely on the orientation of the vectors.

### 5. **Finding the Closest Match**
Given that user input may not always match exactly with the titles in our dataset, we use the **difflib.get_close_matches** function from Python’s standard library. This function helps find the closest match to the user’s input movie title, ensuring that even with slight variations or typos, the system can still provide accurate recommendations.

## 🧠 Building the Model

### 1. **Data Preparation**
The first step involves loading and preparing the movie dataset. The dataset is fetched from a CSV file hosted on GitHub. We focus on five key features:
- **Genres**
- **Keywords**
- **Tagline**
- **Cast**
- **Director**

These features are selected because they are crucial in defining the content of a movie and thus, are likely to influence a user’s preference.

### 2. **Handling Missing Data**
Missing values in the dataset are filled with empty strings to avoid errors during the vectorization process. This step ensures that all features are processed uniformly.

### 3. **Combining Features**
The selected features are combined into a single string for each movie. This combined feature set is then vectorized using the TF-IDF vectorizer. The result is a matrix where each row represents a movie and each column represents a term from the combined features.

### 4. **Calculating Similarity**
The cosine similarity between the feature vectors of all movies is computed. This similarity matrix is the core of the recommendation engine, as it allows us to identify which movies are most similar to a given movie.

### 5. **Generating Recommendations**
When a user inputs their favorite movie, the system finds the closest match in the dataset and retrieves the corresponding index. The similarity scores between this movie and all other movies are then sorted in descending order. The top results are presented as the recommended movies.

## 🔍 Conclusion

The Movie Recommendation System leverages the power of machine learning to provide personalized movie suggestions. By combining content-based filtering, TF-IDF vectorization, and cosine similarity, the system effectively identifies movies that are similar to a user's preferences. This project serves as a practical example of how text-based features can be used to build recommendation systems in real-world applications.

Feel free to explore and modify the code to enhance the recommendation system further. Happy coding and happy watching! 🎬
