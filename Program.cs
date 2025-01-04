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

        (List<int> bestFeatures, float bestError) = KNNOptimizer.DeterministicGreedyFeatureAddition(k: 1, mnistTrain, Error.ArgmaxError, threadCount: 8, verbose: true);

        for (int k = 1; k < 25; k++)
        {
            KNN knn = new KNN(k: k, mnistTrain, bestFeatures);
            float error = Error.ArgmaxError(mnistTest, mnistTest.Select(sample => knn.Predict(sample.input, excludeZero: false)).ToList());
            Console.WriteLine($"k: {k}, Error: {error}");
        }


        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}