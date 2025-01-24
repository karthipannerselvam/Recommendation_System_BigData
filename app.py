from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize SparkSession
spark = SparkSession.builder.appName("ContentBasedAPI").getOrCreate()

# Load Movies Dataset
movies = spark.read.csv("hdfs://localhost:9000/user/hadoop/movies/movies.csv", header=True)
movies = movies.withColumn("genres_list", split(movies["genres"], "\\|"))

# Vectorize Genres
cv = CountVectorizer(inputCol="genres_list", outputCol="genres_features")
model = cv.fit(movies)
movies_features = model.transform(movies)

# Convert Sparse Vectors to Dense Arrays
def sparse_to_dense(sparse_vector):
    return sparse_vector.toArray().tolist()

dense_udf = udf(sparse_to_dense, ArrayType(FloatType()))
movies_features = movies_features.withColumn("genres_dense", dense_udf(movies_features["genres_features"]))

# API Endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get movie title from the request
    movie_title = request.json.get("title", "").lower()
    try:
        # Filter movie by title
        movie_row = movies_features.filter(movies_features["title"].rlike(movie_title)).head(1)
        if not movie_row:
            return jsonify({"error": "Movie not found"}), 404

        # Get genres features of the requested movie
        target_features = movie_row[0]["genres_dense"]

        # Compute similarity with other movies
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude = (sum(a ** 2 for a in vec1) ** 0.5) * (sum(b ** 2 for b in vec2) ** 0.5)
            return dot_product / magnitude if magnitude else 0.0

        movies_with_similarity = movies_features.rdd.map(lambda row: {
            "title": row["title"],
            "similarity": cosine_similarity(row["genres_dense"], target_features)
        }).filter(lambda x: x["similarity"] > 0).takeOrdered(10, key=lambda x: -x["similarity"])

        return jsonify(movies_with_similarity)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
