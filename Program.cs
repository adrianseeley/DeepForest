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
        tw.WriteLine("decay,correct,accuracy");
        List<float> decays = new List<float>();
        for (float decay = 1.0f; decay >= 0f; decay -= 0.001f)
        {
            decays.Add(decay);
        }
        object writeLock = new object();
        Parallel.ForEach(decays, decay =>
        //for (float decay = 0.1f; decay <= 100; decay += 0.1f)
        {
            int correct = 0;
            for (int testIndex = 0; testIndex < normalizedTest.Count; testIndex++)
            {
                Sample testSample = normalizedTest[testIndex];
                
                List<(Sample trainSample, float distance, float preWeight)> trainSamples = new List<(Sample, float, float)>(normalizedTrain.Count);
                for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
                {
                    float distance = testTrainDistances[testIndex, trainIndex];
                    float preWeight = testTrainPreWeights[testIndex, trainIndex];
                    trainSamples.Add((normalizedTrain[trainIndex], distance, preWeight));
                }
                trainSamples.Sort((a, b) => a.distance.CompareTo(b.distance));

                float weightSum = 0f;
                float[] output = new float[outputLength];
                float multiplier = 1f;
                for (int trainIndex = 0; trainIndex < trainSamples.Count; trainIndex++)
                {
                    float weight = trainSamples[trainIndex].preWeight * multiplier;
                    weightSum += weight;
                    for (int i = 0; i < outputLength; i++)
                    {
                        output[i] += trainSamples[trainIndex].trainSample.output[i] * weight;
                    }
                    multiplier *= decay;
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
                Console.WriteLine($"decay={decay} correct={correct} accuracy={accuracy}");
                tw.WriteLine($"{decay},{correct},{accuracy}");
                tw.Flush();
            }
        });

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}