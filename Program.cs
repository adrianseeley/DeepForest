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

        try { Directory.Delete("./bmp", true); } catch { }
        try { Directory.CreateDirectory("./bmp"); } catch { }

        // 2d spiral every 30 points changes color
        List<Sample> samples = new List<Sample>();
        bool color = false;
        for (int i = 0; i < 1000; i++)
        {
            float t = (float)i / 1000.0f * 4.0f * (float)Math.PI;
            float x = (float)Math.Cos(t) * t;
            float y = (float)Math.Sin(t) * t;
            float[] input = new float[] { x, y };
            float[] output = new float[] { color ? 1 : 0, color ? 0 : 1 };
            samples.Add(new Sample(input, output));
            if (i % 30 == 0)
            {
                color = !color;
            }
        }

        List<Sample> normalizedSamples = Sample.Normalize(samples);
        GraphDiffusion graphDiffusion = new GraphDiffusion(normalizedSamples, diffusionRate: 0.1f, diffusionSteps: 10000);

        

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}