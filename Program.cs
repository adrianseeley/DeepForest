﻿using System.Reflection;

public class Program
{
    public static List<Sample> ReadMNIST (string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            float[] labelOneHot = new float[10];
            labelOneHot[labelInt] = 1;
            float[] input = new float[parts.Length - 1];
            for (int i = 1; i < parts.Length; i++)
            {
                input[i - 1] = float.Parse(parts[i]);
            }
            samples.Add(new Sample(input, labelOneHot));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static void Main()
    {
        
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> normalizedTrain, List<Sample> normalizedTest) = Sample.Conormalize(mnistTrain, mnistTest);

        /*
        Model model = new ResidualAggregator(learningRate: 0.01f, verbose: true,
            models: Model.Create(100, () => new BootstrapAggregator(featurePercentPerBag: 0.5f, verbose: false,
                models: Model.Create(100, () => new RandomTree(minSamplesPerLeaf: 5, maxLeafDepth: null, maxSplitAttempts: 100))
            ))
        );*/

        Model model =
        // stacked model
        new StackAggregator(
            verbose: true, 
            models: [
                // concat layers
                .. Model.Create(1, () =>
                    new ConcatAggregator(
                        verbose: true, 
                        models: Model.Create(50, () => 
                            new BootstrapAggregator(
                                featurePercentPerBag: 0.5f, 
                                models: Model.Create(50, () => 
                                    new RandomTree(
                                        minSamplesPerLeaf: 10, 
                                        maxLeafDepth: null, 
                                        maxSplitAttempts: 100
                                    )
                                )
                            )
                        )
                    )
                ),

                // final aggregation layer
                new BootstrapAggregator(
                    verbose: true,
                    featurePercentPerBag: 0.5f,
                    models: Model.Create(50, () =>
                        new RandomTree(
                            minSamplesPerLeaf: 10,
                            maxLeafDepth: null,
                            maxSplitAttempts: 100
                        )
                    )
                )
            ]
        );

        model.Fit(normalizedTrain, Enumerable.Range(0, normalizedTrain[0].input.Length).ToList());
        List<float[]> predictions = model.PredictSamples(normalizedTest);
        float argmaxError = Error.ArgmaxError(normalizedTest, predictions);
        Console.WriteLine($"Argmax Error: {argmaxError}");

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}