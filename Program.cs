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
        TextWriter tw = new StreamWriter("results.csv");
        tw.WriteLine("k,correct,accuracy");
        List<int> ks = new List<int>();
        for (int k = 1; k < 200; k++)
        {
            ks.Add(k);
        }

        Dictionary<int, int> kCorrects = new Dictionary<int, int>();
        foreach (int k in ks)
        {
            kCorrects.Add(k, 0);
        }
        object lockObject = new object();
        int done = 0;
        Parallel.For(0, normalizedTest.Count, testIndex =>
        {
            Sample testSample = normalizedTest[testIndex];
            List<(Sample trainSample, float distance)> trainSampleDistances = new List<(Sample, float)>(normalizedTrain.Count);
            for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                float distance = Utility.EuclideanDistance(testSample.input, normalizedTrain[trainIndex].input);
                trainSampleDistances.Add((normalizedTrain[trainIndex], distance));
            }
            trainSampleDistances.Sort((a, b) => a.distance.CompareTo(b.distance));

            float[] output = new float[outputLength];
            foreach (int k in ks)
            {
                Array.Clear(output);
                float weightSum = 0f;
                for (int trainIndex = 0; trainIndex < k; trainIndex++)
                {
                    Sample trainSample = trainSampleDistances[trainIndex].trainSample;
                    float distance = trainSampleDistances[trainIndex].distance;
                    float weight = 1f / (distance + 0.0000001f);
                    weightSum += weight;
                    for (int i = 0; i < outputLength; i++)
                    {
                        output[i] += trainSample.output[i] * weight;
                    }
                }
                for (int i = 0; i < outputLength; i++)
                {
                    output[i] /= weightSum;
                }
                if (Error.ArgmaxEquals(testSample, output))
                {
                    lock (lockObject)
                    {
                        kCorrects[k]++;
                    }
                }
            }
            lock (lockObject)
            {
                done++;
                Console.WriteLine($"{done}/{normalizedTest.Count}");
            }
        });
        foreach(int k in ks)
        {
            int correct = kCorrects[k];
            float accuracy = (float)correct / (float)normalizedTest.Count;
            tw.WriteLine($"{k},{correct},{accuracy}");
        }
        tw.Flush();


        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}