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
        //List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: -1);
        //List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: -1);
        //(List<Sample> normalizedTrain, List<Sample> normalizedTest) = Sample.Conormalize(mnistTrain, mnistTest);

        try { Directory.Delete("./out", true); } catch (Exception _) { }
        try { Directory.CreateDirectory("./out"); } catch (Exception _) {}

        int width = 256;
        int height = 256;
        Test2D test2D = Test2D.CreateSpiral(1000, 30, 4, width, height);

        for (float exponent = 1f; exponent <= 500f; exponent += 1f)
        {
            Console.WriteLine($"exponent={exponent}");
            List<(float[] input, float[] output)> testPredictions = new List<(float[], float[])>(width * height);
            foreach (float[] input in test2D.testInputs)
            {
                List<(Sample sample, float distance)> sampleDistances = new List<(Sample sample, float distance)>();
                foreach(Sample sample in test2D.normalizedSamples)
                {
                    float distance = Utility.EuclideanDistance(input, sample.input);
                    sampleDistances.Add((sample, distance));
                }
                float maxDistance = sampleDistances.Max(sd => sd.distance);
                maxDistance = Math.Max(maxDistance, 0.0001f);
                List<float> weights = new List<float>();
                foreach ((Sample sample, float distance) in sampleDistances)
                {
                    float weight = MathF.Pow(1f - (distance / maxDistance), exponent);
                    weights.Add(weight);
                }
                float weightSum = weights.Sum();
                int outputLength = sampleDistances[0].sample.output.Length;
                float[] output = new float[outputLength];
                for (int neighbour = 0; neighbour < sampleDistances.Count; neighbour++)
                {
                    for (int i = 0; i < outputLength; i++)
                    {
                        output[i] += sampleDistances[neighbour].sample.output[i] * weights[neighbour];
                    }
                }
                for (int i = 0; i < outputLength; i++)
                {
                    output[i] /= weightSum;
                }
                testPredictions.Add((input, output));
            }
            test2D.Render($"./out/exp_{exponent}.bmp", testPredictions);
        }

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}