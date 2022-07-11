import java.nio.file.{Files, Paths}

import org.apache.commons.lang3.StringUtils
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

import scala.io.Source

object PriceCatalogue {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Price Catalogue")
      .master("local[*]")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", 12)
      .getOrCreate()

    import spark.implicits._

    val csvPath = "data/invoices_sample.csv"
    val stopWordsPath = "data/stopwords.txt"
    val priceCataloguePath = "data/price_catalogue.parquet"

    val invoicesRawDF = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv(csvPath)

    val invoicesDF = invoicesRawDF.transform(filterColumns(_))
      .transform(prepareDescription(_, "description", stopWordsPath))

    val priceCatalogueDF = groupSimilarDescriptions(invoicesDF, priceCataloguePath, 0.15, spark)

    priceCatalogueDF.show(truncate = false)
    priceCatalogueDF.printSchema()

  }

  def filterColumns(df: DataFrame): DataFrame = {

    return df.selectExpr("concat(de_item_in1, ' ', de_item_in2) as description",
      "concat_ws('_', cd_municipio, dt_versao_orc, cd_orgao, trim(cd_unid_orc), nu_nota_empenho, nu_nf) as key",
      "dt_emissao_ne as date",
      "month(dt_emissao_ne) as month",
      "year(dt_emissao_ne) as year",
      "nu_quant_comprada_in as qty",
      "vl_unit_item_in as price")
      .filter(col("description").isNotNull)

  }

  def prepareDescription(df: DataFrame, descriptionCol: String, stopWordsPath: String): DataFrame = {

    // remove punctuation and accents
    val woPunctuationDF = df.withColumn(descriptionCol, trim(lower(regexp_replace(regexp_replace(translate(col(descriptionCol), "ÁÉÍÓÚáéíóúÀÈÌÒÙàèìòùÂÊÎÔÛâêîôûÃẼĨÕŨãẽĩõũÇç", "AEIOUaeiouAEIOUaeiouAEIOUaeiouAEIOUaeiouCc"),"""[^\sa-zA-Z0-9]""", " "), """(\s\s+)""", " "))))

    // remove stop words
    val stopWords = getStopWordsList(stopWordsPath)

    val tokenizer = new Tokenizer()
    .setInputCol(descriptionCol)
    .setOutputCol("words")

    val remover = new StopWordsRemover()
    .setInputCol("words")
    .setOutputCol("words_clean")
    .setStopWords(stopWords)

    // create 5 grams
    val regexTokenizer = new RegexTokenizer()
    .setInputCol("words_clean")
    .setOutputCol("characters")
    .setPattern("")

    val fivegrams = new NGram()
    .setN(5)
    .setInputCol("characters")
    .setOutputCol("five_grams")

    val wordsDF = tokenizer.transform(woPunctuationDF)
    val cleanDF = remover.transform(wordsDF)
      .withColumn("words_clean", concat_ws(" ", col("words_clean")))
    val charactersDF = regexTokenizer.transform(cleanDF)
    val trigramsDF = fivegrams.transform(charactersDF)

    trigramsDF.drop("words", "words_clean", "tokens")

  }

  def groupSimilarDescriptions(df: DataFrame, priceCatalogueFilePath: String, threshold: Double, spark: SparkSession): DataFrame = {

//    val priceCatalogueSchema = StructType(Array(
//      StructField("description", StringType, true),
//      StructField("hash", StringType, true),
//      StructField("key", StringType, true),
//      StructField("tf_vectors", VectorType, true),
//      StructField("qty_similar_items", IntegerType, true),
//      StructField("min_price", DoubleType, true),
//      StructField("max_price", DoubleType, true),
//      StructField("mean_price", DoubleType, true),
//      StructField("median_price", DoubleType, true),
//      StructField("median_price", ArrayType(DoubleType), true)
//    ))
//
//    val priceCatalogueDF = createOrLoadParquetDataframe(priceCatalogueFilePath, priceCatalogueSchema, spark)

    // create a hashing TF vectorizer using the 5-grams to encode the descriptions
    val htfVectorizer = new HashingTF()
    .setInputCol("five_grams")
    .setOutputCol("tf_vectors")
    .setNumFeatures(2^21)

    val vectorizedDF = htfVectorizer.transform(df)

    vectorizedDF.show(truncate = false)
    vectorizedDF.printSchema()

    val mh = new MinHashLSH()
    .setInputCol("tf_vectors")
    .setOutputCol("hash")
    .setNumHashTables(7)

    val mhModel = mh.fit(vectorizedDF)

    val similarityDF = mhModel.approxSimilarityJoin(vectorizedDF, vectorizedDF, threshold)

    val minSimilarKeyDF = similarityDF.selectExpr("datasetA.key as key",
      "datasetB.key as similar_key")
      .groupBy("key")
      .agg(min("similar_key").alias("min_key"))

    //     root
    //     |-- datasetA: struct (nullable = false)
    //     |    |-- description: string (nullable = true)
    //     |    |-- key: string (nullable = false)
    //     |    |-- date: date (nullable = true)
    //     |    |-- month: integer (nullable = true)
    //     |    |-- year: integer (nullable = true)
    //     |    |-- qty: double (nullable = true)
    //     |    |-- price: double (nullable = true)
    //     |    |-- tri-grams: array (nullable = true)
    //     |    |    |-- element: string (containsNull = true)
    //     |    |-- tf_vectors: vector (nullable = true)
    //     |    |-- hash: array (nullable = true)
    //     |    |    |-- element: vector (containsNull = true)
    //     |-- datasetB: struct (nullable = false)
    //     |    |-- description: string (nullable = true)
    //     |    |-- key: string (nullable = false)
    //     |    |-- date: date (nullable = true)
    //     |    |-- month: integer (nullable = true)
    //     |    |-- year: integer (nullable = true)
    //     |    |-- qty: double (nullable = true)
    //     |    |-- price: double (nullable = true)
    //     |    |-- tri-grams: array (nullable = true)
    //     |    |    |-- element: string (containsNull = true)
    //     |    |-- tf_vectors: vector (nullable = true)
    //     |    |-- hash: array (nullable = true)
    //     |    |    |-- element: vector (containsNull = true)
    //     |-- distCol: double (nullable = false)
    val priceCatalogueDF = similarityDF.join(minSimilarKeyDF, similarityDF("datasetA.key") === minSimilarKeyDF("min_key"), "inner")
      .select("min_key",
        "key",
        "distCol",
        "datasetB.price",
        "datasetA.description",
        "datasetB.description")
//    .groupBy("datasetA.description")
//    .agg(
//      collect_list("distCol").alias("similar_items_dist"),
//      first("datasetA.key").alias("key"),
//      first("datasetA.tf_vectors").alias("tf_vectors"),
//      first("datasetA.hash").alias("hash"),
//      count("datasetB").alias("qty_similar_items"),
//      min(col("datasetB.price").cast("double")).alias("min_price"),
//      max(col("datasetB.price").cast("double")).alias("max_price"),
//      avg(col("datasetB.price").cast("double")).alias("mean_price"),
//      percentile_approx(col("datasetB.price").cast("double"), lit(0.5).cast("double"), lit(100).cast("int")).alias("median_price"),
//      collect_list("datasetB.price").alias("price_list"),
//      collect_list("datasetB.description").alias("similar_items")
//    )

    priceCatalogueDF

  }

  private def createOrLoadParquetDataframe(path: String, schema: StructType, spark: SparkSession): DataFrame = {

    if (Files.exists(Paths.get(path)))
      return spark.read.parquet(path)

    val emptyDF = spark.sparkContext.emptyRDD[Row]

    spark.createDataFrame(emptyDF, schema)

  }

  private def getStopWordsList(stopWordsFilePath: String): Array[String] = {

    Source.fromFile(stopWordsFilePath)
      .mkString
      .split(",")
      .map(r => StringUtils.stripAccents(r))

  }

}
