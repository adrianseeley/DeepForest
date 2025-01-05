﻿public class Program
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

        List<Sample> train = [
            .. mnistTrain,
            //.. Synthetic.DropoutFeatures(mnistTrain, 0.1f, 10),
            //.. Synthetic.NoiseFeatures(mnistTrain, 128, 10)
        ];

        ResidualRandomForest rrf = new ResidualRandomForest(train, treeCount: 10000, minSamplesPerLeaf: (int)MathF.Sqrt(train.Count), splitAttempts: 100, learningRate: 0.001f, threadCount: 12, verbose: true);
        
        float error = Error.ArgmaxError(mnistTest, mnistTest.Select(s => rrf.Predict(s.input)).ToList());
        Console.WriteLine($"Error: {error}");

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}