import os

import unidecode
from pyspark.ml.feature import Tokenizer, StopWordsRemover, MinHashLSH, RegexTokenizer, NGram, \
    HashingTF
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType

spark = SparkSession.builder.config("spark.driver.host", "localhost").config("driver-memory", "10G").getOrCreate()

CSV_DATA_PATH = "../data/notas_fiscais_sample.csv"
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
        .transform(lambda df: prepare_description(df, "description", STOP_WORDS_PATH))

    items_df.show(truncate=False)

    # words_dictionary_df = update_words_dictionary(items_df, "description", WORDS_DICTIONARY_PATH)

    # words_dictionary_df.show()

    similarity_df = group_similar_descriptions(items_df, PRICE_CATALOGUE_PATH, 0.4)

    similarity_df.show(n=10, truncate=False)

    similarity_df.printSchema()

    similarity_df.write.parquet(path="data/similarity_df.parquet", mode="overwrite")


def create_or_load_parquet_dataframe(path: str, schema: str) -> DataFrame:
    empty_rdd = spark.sparkContext.emptyRDD()
    df = spark.createDataFrame(data=empty_rdd, schema=schema)

    if os.path.exists(path):
        df = spark.read.parquet(path)

    return df


def filter_columns(df: DataFrame) -> DataFrame:
    return df.selectExpr("concat(de_item_in1, ' ', de_item_in2) as description",
                         "concat_ws('_', cd_municipio, dt_versao_orc, cd_orgao, trim(cd_unid_orc), nu_nota_empenho, nu_nf) as key",
                         "dt_emissao_ne as date",
                         "month(dt_emissao_ne) as month",
                         "year(dt_emissao_ne) as year",
                         "nu_quant_comprada_in as qty",
                         "vl_unit_item_in as price")\
        .filter(F.col("description").isNotNull())


def prepare_description(df: DataFrame, description_col: str, stop_words_file_path: str) -> DataFrame:
    # remove punctuation and accents
    wo_puctuation_df = df.withColumn(description_col, F.trim(F.lower(F.regexp_replace(F.regexp_replace(F.translate(F.col(description_col), "ÁÉÍÓÚáéíóúÀÈÌÒÙàèìòùÂÊÎÔÛâêîôûÃẼĨÕŨãẽĩõũÇç", "AEIOUaeiouAEIOUaeiouAEIOUaeiouAEIOUaeiouCc"),"""[^\sa-zA-Z0-9]""", " "), """(\s\s+)""", " "))))

    # remove stop words
    stop_words = unidecode.unidecode(open(stop_words_file_path, encoding='utf-8').read().lower()).split(",")

    tokenizer = Tokenizer()\
        .setInputCol(description_col)\
        .setOutputCol("words")

    remover = StopWordsRemover()\
        .setInputCol("words")\
        .setOutputCol("words_clean")\
        .setStopWords(stop_words)

    # create 5 grams
    regex_tokenizer = RegexTokenizer()\
        .setInputCol("words_clean")\
        .setOutputCol("characters")\
        .setPattern("")

    trigrams = NGram()\
        .setN(3)\
        .setInputCol("characters")\
        .setOutputCol("n_grams")

    words_df = tokenizer.transform(wo_puctuation_df)
    clean_df = remover.transform(words_df)\
        .withColumn("words_clean", F.concat_ws(" ", F.col("words_clean")))
    characters_df = regex_tokenizer.transform(clean_df)
    n_grams_df = trigrams.transform(characters_df)

    return n_grams_df.drop("words", "words_clean", "tokens")


def group_similar_descriptions(df: DataFrame, price_catalogue_path: str, threshold: float) -> DataFrame:

    price_catalogue_schema = StructType([
        StructField("description", StringType(), True),
        StructField("hash", StringType(), True),
        StructField("key", StringType(), True),
        StructField("tf_vectors", VectorUDT(), True),
        StructField("qty_similar_items", IntegerType(), True),
        StructField("min_price", DoubleType(), True),
        StructField("max_price", DoubleType(), True),
        StructField("mean_price", DoubleType(), True),
        StructField("median_price", DoubleType(), True),
        StructField("median_price", ArrayType(DoubleType()), True)
        ])

    price_catalogue_df = create_or_load_parquet_dataframe(price_catalogue_path, price_catalogue_schema)

    # create a hashing TF vectorizer using the n-grams to encode the descriptions
    htf_vectorizer = HashingTF()\
        .setInputCol("n_grams")\
        .setOutputCol("tf_vectors")\
        .setNumFeatures(2**21)

    vectorized_df = htf_vectorizer.transform(df)

    mh = MinHashLSH()\
        .setInputCol("tf_vectors")\
        .setOutputCol("hash")\
        .setNumHashTables(15)

    mh_model = mh.fit(vectorized_df)

    similarity_df: DataFrame = mh_model.approxSimilarityJoin(vectorized_df, vectorized_df, threshold)

    min_similar_key_df = similarity_df\
        .selectExpr("datasetA.key as key",
                    "datasetB.key as similar_key")\
        .groupBy("key")\
        .agg(F.min("similar_key").alias("min_key"))

    """
     root
     |-- datasetA: struct (nullable = false)
     |    |-- description: string (nullable = true)
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
     |    |-- description: string (nullable = true)
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
    price_catalogue_df = similarity_df.join(min_similar_key_df, similarity_df["datasetA.key"] == min_similar_key_df["min_key"])\
        .groupBy("datasetA.description")\
        .agg(
            F.collect_list("distCol").alias("similar_items_dist"),
            F.first("datasetA.key").alias("key"),
            F.first("datasetA.tf_vectors").alias("tf_vectors"),
            F.first("datasetA.hash").alias("hash"),
            F.count("datasetB").alias("qty_similar_items"),
            F.min(F.col("datasetB.price").cast("double")).alias("min_price"),
            F.max(F.col("datasetB.price").cast("double")).alias("max_price"),
            F.avg(F.col("datasetB.price").cast("double")).alias("mean_price"),
            F.percentile_approx(F.col("datasetB.price").cast("double"), F.lit(0.5).cast("double"), F.lit(100).cast("int")).alias("median_price"),
            F.collect_list("datasetB.price").alias("price_list"),
            F.collect_list("datasetB.description").alias("similar_items")
        ).sort("key")

    return price_catalogue_df


if __name__ == '__main__':
    main()





