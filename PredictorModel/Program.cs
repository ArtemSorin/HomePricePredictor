using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Linq;

namespace PredictorModel
{
    internal class Program
    {
        public class HousesData
        {
            [LoadColumn(17)]
            public float OverallQual;
            [LoadColumn(43)]
            public float firstFlrSF;
            [LoadColumn(46)]
            public float GrLivArea;
            [LoadColumn(61)]
            public float GarageCars;
            [LoadColumn(62)]
            public float GarageArea;
            [LoadColumn(80)]
            public float SalePrice;
        }
        public class HouseDataPrediction
        {
            [ColumnName("PredictedLabel")]
            public float SalePrice;
        }
        [CustomMappingFactoryAttribute("CustomHouseDataMapping")]
        public class MyCustomHouseDataMapping : CustomMappingFactory<HousesData, HousesData>
        {
            public static void CustomAction(HousesData input, HousesData output)
            {
                output.OverallQual = float.Parse(input.OverallQual.ToString());
                output.firstFlrSF = float.Parse(input.firstFlrSF.ToString());
                output.GrLivArea = float.Parse(input.GrLivArea.ToString());
                output.GarageCars = float.Parse(input.GarageCars.ToString());
                output.GarageArea = float.Parse(input.GarageArea.ToString());
                output.SalePrice = float.Parse(input.SalePrice.ToString());
            }

            public override Action<HousesData, HousesData> GetMapping()
            {
                return CustomAction;
            }
        }
        static readonly string _dataPath = @"C:\Users\artem\source\repos\HomePricePredictor\PredictorModel\train.csv";
        static readonly string _modelPath = @"C:\Users\artem\source\repos\HomePricePredictor\PredictorModel\home_price_predictor_model.zip";
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 42);

            IDataView dataView = mlContext.Data.LoadFromTextFile<HousesData>(_dataPath, hasHeader: true, separatorChar: ',');

            var dataProcessPipeline = mlContext.Transforms.CustomMapping((HousesData input, HousesData output) =>
            {
                output.OverallQual = float.Parse(input.OverallQual.ToString());
                output.firstFlrSF = float.Parse(input.firstFlrSF.ToString());
                output.GrLivArea = float.Parse(input.GrLivArea.ToString());
                output.GarageCars = float.Parse(input.GarageCars.ToString());
                output.GarageArea = float.Parse(input.GarageArea.ToString());
                output.SalePrice = float.Parse(input.SalePrice.ToString());
            }, contractName: "CustomHouseDataMapping")
            .Append(mlContext.Transforms.ReplaceMissingValues(new[] {
                new InputOutputColumnPair("OverallQual", "OverallQual"),
                new InputOutputColumnPair("firstFlrSF", "firstFlrSF"),
                new InputOutputColumnPair("GrLivArea", "GrLivArea"),
                new InputOutputColumnPair("GarageCars", "GarageCars"),
                new InputOutputColumnPair("GarageArea", "GarageArea"),
                new InputOutputColumnPair("SalePrice", "SalePrice"),
            }))
            .Append(mlContext.Transforms.Concatenate("Features", "OverallQual", "firstFlrSF", "GrLivArea", "GarageCars",
            "GarageArea"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "SalePrice"))
            .AppendCacheCheckpoint(mlContext);

            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3);
            IDataView trainingData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            var trainingPipeline = dataProcessPipeline
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    binaryEstimator: mlContext.BinaryClassification.Trainers.FastForest(
                        labelColumnName: "Label",
                        featureColumnName: "Features"),
                    labelColumnName: "Label"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            IDataView predictions = trainedModel.Transform(testData);

            mlContext.Model.Save(trainedModel, trainingData.Schema, _modelPath);

            ITransformer mlModel = mlContext.Model.Load(_modelPath, out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<HousesData, HouseDataPrediction>(mlModel);

            HousesData newHouseData = new HousesData
            {
                OverallQual = 7,
                firstFlrSF = 856,
                GrLivArea = 1710,
                GarageCars = 2,
                GarageArea = 548,
            };

            HouseDataPrediction prediction = predEngine.Predict(newHouseData);

            Console.WriteLine($"Predicted SalePrice: {prediction.SalePrice}");
            Console.ReadLine();
        }
    }
}
