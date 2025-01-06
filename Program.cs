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

        TextWriter tw = new StreamWriter("./results.csv", append: false);
        tw.WriteLine("Resample,Error");

        List<bool> configs = new List<bool>();
        configs.Add(false);
        configs.Add(true);
      
        ParallelOptions po = new ParallelOptions();
        po.MaxDegreeOfParallelism = 12;
        object writeLock = new object();
        for (int i = 0; i < 100; i++)
        Parallel.ForEach(configs, po, config => {
            bool testFlag = config;
            int testInt = testFlag ? 1 : 0;
            ResidualRandomForest rrf = new ResidualRandomForest(mnistTrain, treeCount: 10000, minSamplesPerLeaf: 15, splitAttempts: 100, learningRate: 0.001f, resample: testFlag, verbose: false);
            float error = Error.ArgmaxError(mnistTest, mnistTest.Select(s => rrf.Predict(s.input)).ToList());
            lock (writeLock)
            {
                tw.WriteLine($"{testInt},{error}");
                tw.Flush();
                Console.WriteLine($"tr: {testInt}, er: {error}");
            }
        });

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}