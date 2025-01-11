using System.Net.Http.Headers;
using System.Reflection;

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
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: -1);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: -1);
        (List<Sample> normalizedTrain, List<Sample> normalizedTest) = Sample.Conormalize(mnistTrain, mnistTest);

        int outputLength = normalizedTrain[0].output.Length;
        float[,] testTrainDistances = new float[normalizedTest.Count, normalizedTrain.Count];
        for (int testIndex = 0; testIndex < normalizedTest.Count; testIndex++)
        {
            Console.Write($"\rDistance {testIndex + 1}/{normalizedTest.Count}");
            Parallel.For(0, normalizedTrain.Count, trainIndex =>
            {
                float distance = Utility.EuclideanDistance(normalizedTest[testIndex].input, normalizedTrain[trainIndex].input);
                testTrainDistances[testIndex, trainIndex] = distance;
            });
            /*for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                float distance = Utility.EuclideanDistance(normalizedTest[testIndex].input, normalizedTrain[trainIndex].input);
                testTrainDistances[testIndex, trainIndex] = distance;
            }*/
        }
        Console.WriteLine();
        float[] testMaxDistances = new float[normalizedTest.Count];
        for (int testIndex = 0; testIndex < normalizedTest.Count; testIndex++)
        {
            Console.Write($"\rMax Distance {testIndex + 1}/{normalizedTest.Count}");
            float maxDistance = testTrainDistances[testIndex, 0];
            for (int trainIndex = 1; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                maxDistance = Math.Max(maxDistance, testTrainDistances[testIndex, trainIndex]);
            }
            testMaxDistances[testIndex] = maxDistance;
        }
        Console.WriteLine();
        float[,] testTrainPreWeights = new float[normalizedTest.Count, normalizedTrain.Count];
        for (int testIndex = 0; testIndex < normalizedTest.Count; testIndex++)
        {
            Console.Write($"\rPre Weight {testIndex + 1}/{normalizedTest.Count}");
            for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                float preWeight = 1f - (testTrainDistances[testIndex, trainIndex] / testMaxDistances[testIndex]);
                testTrainPreWeights[testIndex, trainIndex] = preWeight;
            }
        }
        Console.WriteLine();

        TextWriter tw = new StreamWriter("results.csv");
        tw.WriteLine("exponent,correct,accuracy");
        List<float> exponents = new List<float>();
        for (float exponent = 0.01f; exponent <= 100f; exponent += 0.01f)
        {
            exponents.Add(exponent);
        }
        object writeLock = new object();
        Parallel.ForEach(exponents, exponent =>
        //for (float exponent = 0.1f; exponent <= 100; exponent += 0.1f)
        {
            int correct = 0;
            for (int testIndex = 0; testIndex < normalizedTest.Count; testIndex++)
            {
                Sample testSample = normalizedTest[testIndex];
                float weightSum = 0f;
                float[] output = new float[outputLength];
                for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
                {
                    float weight = MathF.Pow(testTrainPreWeights[testIndex, trainIndex], exponent);
                    weightSum += weight;
                    for (int i = 0; i < outputLength; i++)
                    {
                        output[i] += normalizedTrain[trainIndex].output[i] * weight;
                    }
                }
                for (int i = 0; i < outputLength; i++)
                {
                    output[i] /= weightSum;
                }
                correct += Error.ArgmaxEquals(testSample, output) ? 1 : 0;
            }
            float accuracy = (float)correct / (float)normalizedTest.Count;
            lock (writeLock)
            {
                Console.WriteLine($"exponent={exponent} correct={correct} accuracy={accuracy}");
                tw.WriteLine($"{exponent},{correct},{accuracy}");
                tw.Flush();
            }
        });

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}