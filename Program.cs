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

        try { Directory.Delete("./frames", true); } catch (Exception e) {}
        try { Directory.CreateDirectory("./frames"); } catch (Exception e) { }
        int width = 2048;
        int height = 2048;
        Test2D test2D = Test2D.CreateSpiral(1000, 30, 4, width, height);
        float[,] normalizedDistances = Interpolator.ComputeNormalizedDistances(test2D.normalizedSamples, test2D.testInputs, verbose: true);
        float[,] testPredictions = new float[test2D.testInputs.Count, test2D.normalizedSamples[0].output.Length];

        int i = 0;
        float closeWeight = 1f;
        float farWeight = 0.01f;
        for (float hingePoint = 0.001f; hingePoint <= 1.0; hingePoint += 0.001f)
        { 
            Console.WriteLine($"exp {hingePoint}");
            Parallel.For(0, test2D.testInputs.Count, testIndex =>
            {
                Interpolator.ComputeHinge(test2D.normalizedSamples, normalizedDistances, testIndex, hingePoint: hingePoint, closeWeight: closeWeight, farWeight: farWeight, testPredictions);
            });
            test2D.Render($"./frames/{i.ToString().PadLeft(4, '0')}.bmp", testPredictions);
            i++;
        }
       

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}