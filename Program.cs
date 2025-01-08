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
        /*
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 10000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> normalizedTrain, List<Sample> normalizedTest) = Sample.Conormalize(mnistTrain, mnistTest);
        */



        int models = 10;
        List<int> allFeatures = [0, 1];
        ProgressionTest2D bpknnTest = ProgressionTest2D.CreateSpiral(1000, 30, 4f, $"./bpknn_{models}.mp4", fps: 1, width: 1280, height: 1280);
        BaggedPredictorKNN bpknn = new BaggedPredictorKNN(bpknnTest.normalizedSamples, 2, k: 3, distanceDelegate: Utility.EuclideanDistance);
        for (int modelIndex = 0; modelIndex < models; modelIndex++)
        {
            Console.WriteLine($"Model {modelIndex + 1}/{models}");
            bpknn.AddModel();
            bpknnTest.Frame(bpknn.Predict);
        }
        bpknnTest.Finish();
        return;

        /*
        ProgressionTest2D bprtTest = ProgressionTest2D.CreateSpiral(1000, 30, 4f, $"./bprt_{models}.mp4", fps: 1, width: 1280, height: 1280);
        ProgressionTest2D bpstTest = ProgressionTest2D.CreateSpiral(1000, 30, 4f, $"./bpst_{models}.mp4", fps: 1, width: 1280, height: 1280);
        ProgressionTest2D rprtTest = ProgressionTest2D.CreateSpiral(1000, 30, 4f, $"./rprt_{models}.mp4", fps: 1, width: 1280, height: 1280);
        ProgressionTest2D rpstTest = ProgressionTest2D.CreateSpiral(1000, 30, 4f, $"./rpst_{models}.mp4", fps: 1, width: 1280, height: 1280);
        BaggedPredictorRandomTree bprt = new BaggedPredictorRandomTree(bprtTest.normalizedSamples, minSamplesPerLeaf: 1, maxLeafDepth: -1, maxSplitAttempts: 100);
        BaggedPredictorStandardTree bpst = new BaggedPredictorStandardTree(bpstTest.normalizedSamples, minSamplesPerLeaf: 1, maxLeafDepth: -1, minimize: StandardTree.Minimize.MeanSquaredError);
        ResidualPredictorRandomTree rprt = new ResidualPredictorRandomTree(rprtTest.normalizedSamples, learningRate: 0.01f, minSamplesPerLeaf: 1, maxLeafDepth: -1, maxSplitAttempts: 100);
        ResidualPredictorStandardTree rpst = new ResidualPredictorStandardTree(rpstTest.normalizedSamples, learningRate: 0.01f, minSamplesPerLeaf: 1, maxLeafDepth: -1, minimize: StandardTree.Minimize.MeanSquaredError);
        for (int modelIndex = 0; modelIndex < models; modelIndex++)
        {
            Console.WriteLine($"Model {modelIndex + 1}/{models}");
            Task bprtTask = new Task(() =>
            {
                bprt.AddModel();
                bprtTest.Frame(bprt.Predict);
            });
            Task bpstTask = new Task(() =>
            {
                bpst.AddModel();
                bpstTest.Frame(bpst.Predict);
            });
            Task rprtTask = new Task(() =>
            {
                rprt.AddModel();
                rprtTest.Frame(rprt.Predict);
            });
            Task rpstTask = new Task(() =>
            {
                rpst.AddModel();
                rpstTest.Frame(rpst.Predict);
            });
            bprtTask.Start();
            bpstTask.Start();
            rprtTask.Start();
            rpstTask.Start();
            Task.WaitAll(bprtTask, bpstTask, rprtTask, rpstTask);
        }
        bprtTest.Finish();
        bpstTest.Finish();
        rprtTest.Finish();
        rpstTest.Finish();
        */

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}