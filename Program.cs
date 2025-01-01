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

    public static float Fitness(List<Sample> samples, List<float[]> predictions)
    {
        int correct = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            float[] actual = samples[i].output;
            float[] prediction = predictions[i];
            float maxValueActual = float.NegativeInfinity;
            int maxValueActualIndex = -1;
            float maxValuePrediction = float.NegativeInfinity;
            int maxValuePredictionIndex = -1;
            for (int j = 0; j < actual.Length; j++)
            {
                if (actual[j] > maxValueActual)
                {
                    maxValueActual = actual[j];
                    maxValueActualIndex = j;
                }
                if (prediction[j] > maxValuePrediction)
                {
                    maxValuePrediction = prediction[j];
                    maxValuePredictionIndex = j;
                }
            }
            if (maxValueActualIndex == maxValuePredictionIndex)
            {
                correct++;
            }
        }
        return (float)correct / (float)samples.Count;
    }

    public static void Main()
    {
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv");
        int inputComponentCount = mnistTrain[0].input.Length;

        Random random = new Random();

        TextWriter tw = new StreamWriter("./results.csv", false);
        tw.WriteLine("minSamplesPerLeaf,nodeCount,treeCount,fitness");
        object writeLock = new object();

        RandomForest rf = new RandomForest(mnistTrain, xComponentCount: 50, treeCount: 0, minSamplesPerLeaf: 1);
        for (int treeIndex = 0; treeIndex < 500; treeIndex++)
        {
            rf.AddTree(mnistTrain);
            List<float[]> predictions = new List<float[]>(mnistTest.Count);
            foreach (Sample sample in mnistTest)
            {
                predictions.Add(rf.Predict(sample.input));
            }
            float fitness = Fitness(mnistTest, predictions);
            long nodeCount = rf.CountNodes();
            lock (writeLock)
            {
                Console.WriteLine($"mspl: {1}, nc: {nodeCount} t: {rf.randomTrees.Count}, f: {fitness}");
                tw.WriteLine($"{1},{nodeCount},{rf.randomTrees.Count},{fitness}");
                tw.Flush();
            }
        }
        long targetNodeCount = rf.CountNodes();

        Parallel.For(2, 100, i =>
        {
            int minSamplesPerLeaf = i;
            RandomForest rf = new RandomForest(mnistTrain, xComponentCount: 50, treeCount: 0, minSamplesPerLeaf: minSamplesPerLeaf);
            for (; ;)
            {
                rf.AddTree(mnistTrain);
                List<float[]> predictions = new List<float[]>(mnistTest.Count);
                foreach (Sample sample in mnistTest)
                {
                    predictions.Add(rf.Predict(sample.input));
                }
                float fitness = Fitness(mnistTest, predictions);
                long nodeCount = rf.CountNodes();
                lock (writeLock)
                {
                    Console.WriteLine($"mspl: {minSamplesPerLeaf}, nc: {nodeCount}, t: {rf.randomTrees.Count}, f: {fitness}");
                    tw.WriteLine($"{minSamplesPerLeaf},{nodeCount},{rf.randomTrees.Count},{fitness}");
                }
                if (nodeCount >= targetNodeCount)
                {
                    break;
                }
            }
        });
        Console.ReadLine();
    }
}