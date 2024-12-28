public class Program
{
    public static List<(List<float> input, List<float> output)> ReadMNIST (string filename, int max = -1)
    {
        List<(List<float> input, List<float> output)> data = new List<(List<float> input, List<float> output)>();
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
            List<float> labelOneHot = new List<float>();
            for (int i = 0; i <= 9; i++)
            {
                if (i == labelInt)
                {
                    labelOneHot.Add(1);
                }
                else
                {
                    labelOneHot.Add(0);
                }
            }
            List<float> input = new List<float>();
            for (int i = 1; i < parts.Length; i++)
            {
                input.Add(float.Parse(parts[i]));
            }
            data.Add((input, labelOneHot));
            if (max != -1 && data.Count >= max)
            {
                break;
            }
        }
        return data;
    }

    public static float OneHotFitness(MultiDeepRandomForest mdrf, List<(List<float> input, List<float> output)> test, bool verbose)
    {
        object trackLock = new object();
        int correct = 0;
        int incorrect = 0;
        Parallel.For(0, test.Count, (index) =>
        {
            int localIndex = index;
            (List<float> input, List<float> output) sample = test[localIndex];
            List<float> predictOutput = mdrf.Predict(sample.input);
            int predictLabel = predictOutput.IndexOf(predictOutput.Max());
            int actualLabel = sample.output.IndexOf(sample.output.Max());
            lock (trackLock)
            {
                if (predictLabel == actualLabel)
                {
                    correct++;
                }
                else
                {
                    incorrect++;
                }
                if (verbose)
                {
                    Console.Write($"\rCorrect: {correct}, Incorrect: {incorrect}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return ((float)correct) / ((float)test.Count);
    }

    public static void Main()
    {
        List<(List<float> input, List<float> output)> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<(List<float> input, List<float> output)> mnistTest = ReadMNIST("D:/data/mnist_test.csv");

        using TextWriter results = new StreamWriter("results.csv");
        results.WriteLine("layers,forests,trees,fitness");

        MultiDeepRandomForest mdrf = new MultiDeepRandomForest(mnistTrain, treesPerForest: 50, forestsPerLayer: 50, layers: 3, extraRandom: true, verbose: true);
        float fitness = OneHotFitness(mdrf, mnistTest, verbose: true);
        Console.WriteLine($"Fitness: {fitness}");
        Console.ReadLine();
    }
}