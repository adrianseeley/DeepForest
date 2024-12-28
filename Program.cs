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

    public static float OneHotFitness(MultiDeepRandomForest mdrf, List<(List<float> input, List<float> output)> test)
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
            }
        });
        return ((float)correct) / ((float)test.Count);
    }

    public static void Main()
    {
        List<(List<float> input, List<float> output)> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<(List<float> input, List<float> output)> mnistTest = ReadMNIST("D:/data/mnist_test.csv");

        using TextWriter results = new StreamWriter("results.csv");
        results.WriteLine("forestCount,treeCount,fitness");

        int[] treeCounts = [1, 5, 10, 50, 100, 500, 1000];

        MultiDeepRandomForest mdrf;
        for (int forestCount = 1; forestCount < 100; forestCount++)
        {
            foreach (int treeCount in treeCounts)
            {
                Console.Write($"F: {forestCount}, T: {treeCount}, R: ");
                mdrf = new MultiDeepRandomForest(mnistTrain, treeCount, forestCount, true);
                float fitness = OneHotFitness(mdrf, mnistTest);
                Console.WriteLine(fitness);
                results.WriteLine($"{forestCount},{treeCount},{fitness}");
                results.Flush();
            }
        }
    }
}