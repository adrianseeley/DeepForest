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

    public static float OneHotFitness(MultiDeepRandomForest mdrf, List<Sample> test, bool verbose)
    {
        object trackLock = new object();
        int correct = 0;
        int incorrect = 0;
        Parallel.For(0, test.Count, (index) =>
        {
            int localIndex = index;
            Sample sample = test[localIndex];
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
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv");

        Random random = new Random();
        BoostedRandomTree brt = new BoostedRandomTree(random, mnistTrain, Enumerable.Range(0, 784).ToList(), 100, 0.1f);
    }
}