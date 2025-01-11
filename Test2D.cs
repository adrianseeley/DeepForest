public class Test2D
{
    public delegate float[] PredictFunction(float[] input);
    public int width;
    public int height;
    public byte[,,] pixels;
    public List<Sample> normalizedSamples;
    public List<float[]> testInputs;

    public static Test2D CreateSpiral(int count, int alternator, float swirls, int width, int height)
    {
        // 2d spiral every alternator points changes color r->g->b
        List<Sample> samples = new List<Sample>();
        int color = 0;
        for (int i = 0; i < count; i++)
        {
            float t = (float)i / (float)count * swirls * MathF.PI;
            float x = MathF.Cos(t) * t;
            float y = MathF.Sin(t) * t;
            float[] input = new float[] { x, y };
            float[] output = new float[] { color == 0 ? 1 : 0, color == 1 ? 1 : 0, color == 2 ? 1 : 0 };
            samples.Add(new Sample(input, output));
            if (i % alternator == 0)
            {
                color++;
                if (color > 2)
                {
                    color = 0;
                }
            }
        }
        List<Sample> normalizedSamples = Sample.Normalize(samples);
        return new Test2D(width, height, normalizedSamples);
    }

    public Test2D(int width, int height, List<Sample> normalizedSamples)
    {
        this.width = width;
        this.height = height;
        this.pixels = Bitmap.Create(width, height);
        this.normalizedSamples = normalizedSamples;
        this.testInputs = new List<float[]>(width * height);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                float nx = (float)x / (float)width;
                float ny = (float)y / (float)height;
                testInputs.Add(new float[] { nx, ny });
            }
        }
    }

    public void Render(List<(float[], float[])> testPredictions)
    {
        // draw predictions
        foreach((float[] input, float[] output) in testPredictions)
        {
            int x = (int)(input[0] * width);
            int y = (int)(input[1] * height);
            byte r = (byte)(output[0] * 255);
            byte g = (byte)(output[1] * 255);
            byte b = (byte)(output[2] * 255);
            Bitmap.DrawPixel(pixels, x, y, r, g, b);
        }

        // draw sample backs
        foreach (Sample normalizedSample in normalizedSamples)
        {
            int x = (int)(normalizedSample.input[0] * width);
            int y = (int)(normalizedSample.input[1] * height);
            Bitmap.FillRect(pixels, x - 2, y - 2, 5, 5, 0, 0, 0);
        }

        // draw samples
        foreach (Sample normalizedSample in normalizedSamples)
        {
            int x = (int)(normalizedSample.input[0] * width);
            int y = (int)(normalizedSample.input[1] * height);
            byte r = (byte)(normalizedSample.output[0] * 255);
            byte g = (byte)(normalizedSample.output[1] * 255);
            byte b = (byte)(normalizedSample.output[2] * 255);
            Bitmap.FillRect(pixels, x - 1, y - 1, 3, 3, r, g, b);
        }
    }

    public void Render(string filename, List<(float[], float[])> testPredictions)
    {
        Render(testPredictions);
        Bitmap.SaveToBMP(filename, pixels);
    }
}