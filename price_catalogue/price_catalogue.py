import os
import sys

import numpy as np
import unidecode
from pyspark.ml.feature import Tokenizer, StopWordsRemover, MinHashLSH, RegexTokenizer, NGram, \
    HashingTF
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, FloatType

spark = SparkSession.builder.config("spark.driver.host", "localhost").config("driver-memory", "10G").getOrCreate()

# fix pycharm error: https://stackoverflow.com/questions/68705417/pycharm-error-java-io-ioexception-cannot-run-program-python3-createprocess
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

CSV_DATA_PATH = "../data/notas_fiscais_amostra.csv"
CSV_DATA_PATH_10pc = "../data/notas_fiscais_10pc.csv"
STOP_WORDS_PATH = "../resources/stopwords.txt"
PRICE_CATALOGUE_PATH = "../data/price_catalogue.parquet"
SCHEMA = 'cd_municipio string,' \
         'dt_versao_orc string,' \
         'cd_orgao string,' \
         'cd_unid_orc string,' \
         'dt_emissao_ne date,' \
         'nu_nota_empenho string,' \
         'dt_liquid_liq date,' \
         'tp_nf string,' \
         'nu_nf string,' \
         'nu_item_seq_in string,' \
         'de_item_in1 string,' \
         'de_item_in2 string,' \
         'de_unid_compra_in string,' \
         'nu_quant_comprada_in double,' \
         'vl_unit_item_in double,' \
         'vl_total_item_in double,' \
         'cd_ncm_in string'


def main():
    print(spark.sparkContext.getConf().getAll())

    raw_df = spark.read.csv(path=CSV_DATA_PATH, header=True, schema=SCHEMA)

    items_df = raw_df.transform(lambda df: filter_columns(df)) \
        .transform(lambda df: prepare_description(df, "description", STOP_WORDS_PATH))\
        .transform(lambda df: compute_n_grams(df, 3))

    items_df.show(truncate=False)

    similarity_flat_df, similarity_df = group_similar_descriptions(items_df, 0.35)

    similarity_df.show(n=1500, truncate=False)

    similarity_flat_df.show(n=1500, truncate=False)

    similarity_flat_df.printSchema()

    overpriced_items_df = find_overpriced_items(similarity_flat_df, similarity_df)

    overpriced_items_df.show(n=1500, truncate=False)

    # similarity_df.write.parquet(path="data/similarity_df.parquet", mode="overwrite")


def find_overpriced_items(df: DataFrame, price_catalogue_df: DataFrame) -> DataFrame:

    price_catalogue_median_df = price_catalogue_df.select("description_hash_key", "median_price")

    overpriced_items_df = df.join(price_catalogue_median_df, df["description_hash_key"] == price_catalogue_median_df["description_hash_key"])\
        .filter("price > 1.5 * median_price")

    return overpriced_items_df

def create_or_load_parquet_dataframe() -> DataFrame:

    path = PRICE_CATALOGUE_PATH

    if os.path.exists(path):
        df = spark.read.parquet(path)
    else:
        price_catalogue_schema = StructType([
            StructField("key", StringType(), True),
            StructField("tf_vectors", VectorUDT(), True),
            StructField("hash", StringType(), True),
            StructField("description", StringType(), True),
            StructField("price_list", ArrayType(DoubleType()), True),
            StructField("similar_keys", ArrayType(StringType()), True),
            StructField("min_price", DoubleType(), True),
            StructField("max_price", DoubleType(), True),
            StructField("mean_price", DoubleType(), True),
            StructField("median_price", DoubleType(), True),
            StructField("qty_similar_items", IntegerType(), True)
        ])

        empty_rdd = spark.sparkContext.emptyRDD()
        df = spark.createDataFrame(data=empty_rdd, schema=price_catalogue_schema)

    return df


def filter_columns(df: DataFrame) -> DataFrame:
    return df.selectExpr("concat_ws(' ', de_item_in1, de_item_in2) as description",
                         "de_unid_compra_in as unit",
                         "concat_ws('_', cd_municipio, dt_versao_orc, cd_orgao, trim(cd_unid_orc), nu_nota_empenho, nu_nf, nu_item_seq_in) as key",
                         "dt_emissao_ne as date",
                         "month(dt_emissao_ne) as month",
                         "year(dt_emissao_ne) as year",
                         "nu_quant_comprada_in as qty",
                         "vl_unit_item_in as price")


def prepare_description(df: DataFrame, description_col: str, stop_words_file_path: str) -> DataFrame:

    # remove punctuation and accents
    wo_punctuation_df = df.filter(f"{description_col} is not null")\
        .withColumn(description_col, F.trim(F.lower(F.regexp_replace(F.regexp_replace(
        F.translate(F.col(description_col), "ÁÉÍÓÚáéíóúÀÈÌÒÙàèìòùÂÊÎÔÛâêîôûÃẼĨÕŨãẽĩõũÇç",
                    "AEIOUaeiouAEIOUaeiouAEIOUaeiouAEIOUaeiouCc"), """[^\sa-zA-Z0-9]""", " "), """(\s\s+)""", " "))))

    # remove stop words
    stop_words = unidecode.unidecode(open(stop_words_file_path, encoding='utf-8').read().lower()).split(",")

    tokenizer = Tokenizer() \
        .setInputCol(description_col) \
        .setOutputCol("words")

    remover = StopWordsRemover() \
        .setInputCol("words") \
        .setOutputCol("words_clean") \
        .setStopWords(stop_words)

    words_df = tokenizer.transform(wo_punctuation_df)
    clean_df = remover.transform(words_df) \
        .withColumn("words_clean", F.concat_ws(" ", F.col("words_clean"))) \
        .filter(F.length(F.trim(F.col("words_clean"))) >= 3)  # remove descriptions with length <= 3

    # create a hash using the clean description
    prepared_df = clean_df.withColumn("description_hash", F.hash(F.col("words_clean")))

    return prepared_df


def compute_n_grams(clean_df: DataFrame, n: int) -> DataFrame:

    # create 5 grams
    regex_tokenizer = RegexTokenizer()\
        .setInputCol("words_clean")\
        .setOutputCol("characters")\
        .setPattern("")

    trigrams = NGram()\
        .setN(n)\
        .setInputCol("characters")\
        .setOutputCol("n_grams")

    characters_df = regex_tokenizer.transform(clean_df)
    n_grams_df = trigrams.transform(characters_df)\
        .drop("words", "words_clean", "tokens", "characters")

    return n_grams_df


def new_items(df: DataFrame,
              price_catalogue_df: DataFrame) -> DataFrame:

    price_catalogue_keys_df = price_catalogue_df.select(F.explode(F.col("similar_keys")).alias("key")).join(df, price_catalogue_df["key"])


def group_similar_descriptions(df: DataFrame,
                               threshold: float,
                               save_model: bool = False,
                               load_model: bool = False) -> DataFrame:

    price_catalogue_df = create_or_load_parquet_dataframe()

    htf_model_path = "models/htf"
    mh_model_path = "models/mh_lsh"

    # create a hashing TF vectorizer using the n-grams to encode the descriptions
    htf_vectorizer = HashingTF().load(htf_model_path) if load_model else HashingTF()\
        .setInputCol("n_grams")\
        .setOutputCol("tf_vectors")\
        .setNumFeatures(2**18)

    mh = MinHashLSH().load(mh_model_path) if load_model else MinHashLSH()\
        .setInputCol("tf_vectors")\
        .setOutputCol("hash")\
        .setNumHashTables(5)

    if save_model:
        htf_vectorizer.save(htf_model_path)
        mh.save(mh_model_path)

    vectorized_df = htf_vectorizer.transform(df)

    mh_model = mh.fit(vectorized_df)

    similarity_df: DataFrame = mh_model.approxSimilarityJoin(vectorized_df, vectorized_df, threshold)

    min_similar_description_hash_df = similarity_df\
        .selectExpr("datasetA.description_hash",
                    "datasetB.description_hash as similar_description_hash")\
        .groupBy("description_hash")\
        .agg(F.min("similar_description_hash").alias("min_description_hash"))\
        .select("min_description_hash")\
        .distinct()

    min_similar_description_hash_df.show(n=50, truncate=False)

    """
     root
     |-- datasetA: struct (nullable = false)
     |    |-- description_hash: string (nullable = true)
     |    |-- description: string (nullable = true)
     |    |-- unit: string (nullable = true)     
     |    |-- key: string (nullable = false)
     |    |-- date: date (nullable = true)
     |    |-- month: integer (nullable = true)
     |    |-- year: integer (nullable = true)
     |    |-- qty: double (nullable = true)
     |    |-- price: double (nullable = true)
     |    |-- tri-grams: array (nullable = true)
     |    |    |-- element: string (containsNull = true)
     |    |-- tf_vectors: vector (nullable = true)
     |    |-- hash: array (nullable = true)
     |    |    |-- element: vector (containsNull = true)
     |-- datasetB: struct (nullable = false)
     |    |-- description_hash: string (nullable = true)
     |    |-- description: string (nullable = true)
     |    |-- unit: string (nullable = true)     
     |    |-- key: string (nullable = false)
     |    |-- date: date (nullable = true)
     |    |-- month: integer (nullable = true)
     |    |-- year: integer (nullable = true)
     |    |-- qty: double (nullable = true)
     |    |-- price: double (nullable = true)
     |    |-- tri-grams: array (nullable = true)
     |    |    |-- element: string (containsNull = true)
     |    |-- tf_vectors: vector (nullable = true)
     |    |-- hash: array (nullable = true)
     |    |    |-- element: vector (containsNull = true)
     |-- distCol: double (nullable = false)
    """
    price_catalogue_flat_df = similarity_df\
        .join(min_similar_description_hash_df, similarity_df["datasetA.description_hash"] == min_similar_description_hash_df["min_description_hash"])\
        .selectExpr("min_description_hash as description_hash_key",
                    "datasetB.key as key",
                    "datasetB.description_hash as description_hash",
                    "datasetB.description as description",
                    "datasetB.unit as unit",
                    "datasetB.date as date",
                    "datasetB.month as month",
                    "datasetB.year as year",
                    "datasetB.qty as qty",
                    "datasetB.price as price",
                    "datasetB.tf_vectors as tf_vectors",
                    "datasetA.hash as hash",
                    "datasetA.unit as reference_unit",
                    "distCol as dist"
                    )\
        .orderBy("description_hash_key")

    array_min = F.udf(lambda x: float(np.min(x)), FloatType())
    array_max = F.udf(lambda x: float(np.max(x)), FloatType())
    array_mean = F.udf(lambda x: float(np.mean(x)), FloatType())
    array_median = F.udf(lambda x: float(np.median(x)), FloatType())

    price_catalogue_df = similarity_df\
        .join(min_similar_description_hash_df, similarity_df["datasetA.description_hash"] == min_similar_description_hash_df["min_description_hash"])\
        .groupBy("datasetA.description_hash")\
        .agg(
            F.first("datasetA.tf_vectors").alias("tf_vectors"),
            F.first("datasetA.hash").alias("hash"),
            F.first("datasetA.description").alias("description"),
            F.collect_list("datasetB.price").alias("price_list"),
            F.collect_list("datasetB.key").alias("similar_keys"),
        )\
        .withColumn("min_price", array_min("price_list")) \
        .withColumn("max_price", array_max("price_list")) \
        .withColumn("mean_price", array_mean("price_list")) \
        .withColumn("median_price", array_median("price_list")) \
        .withColumn("qty_similar_items", F.size("price_list"))\
        .withColumnRenamed("description_hash", "description_hash_key")

    return price_catalogue_flat_df, price_catalogue_df


if __name__ == '__main__':
    main()





