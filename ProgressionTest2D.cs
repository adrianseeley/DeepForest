public class ProgressionTest2D
{
    public delegate float[] PredictFunction(float[] input);
    public int width;
    public int height;
    public byte[,,] pixels;
    public FFMPEG ffmpeg;
    public List<Sample> normalizedSamples;

    public static ProgressionTest2D CreateSpiral(int count, int alternator, float swirls, string filename, int fps, int width, int height)
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
        return new ProgressionTest2D(filename, fps, width, height, normalizedSamples);
    }

    public ProgressionTest2D(string filename, int fps, int width, int height, List<Sample> normalizedSamples)
    {
        this.width = width;
        this.height = height;
        this.pixels = Bitmap.Create(width, height);
        this.ffmpeg = new FFMPEG(filename, fps, width, height, FFMPEG.EncodeAs.MP4);
        this.normalizedSamples = normalizedSamples;
    }

    public void Frame(PredictFunction predictFunction)
    {
        // draw predictions
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                float nx = (float)x / (float)width;
                float ny = (float)y / (float)height;
                float[] prediction = predictFunction([nx, ny]);
                byte r = (byte)(prediction[0] * 255);
                byte g = (byte)(prediction[1] * 255);
                byte b = (byte)(prediction[2] * 255);
                Bitmap.DrawPixel(pixels, x, y, r, g, b);
            }
        }

        // draw samples
        foreach (Sample normalizedSample in normalizedSamples)
        {
            int x = (int)(normalizedSample.input[0] * width);
            int y = (int)(normalizedSample.input[1] * height);
            byte r = (byte)(normalizedSample.output[0] * 255);
            byte g = (byte)(normalizedSample.output[1] * 255);
            byte b = (byte)(normalizedSample.output[2] * 255);
            Bitmap.DrawPixel(pixels, x, y, r, g, b);
        }

        ffmpeg.AddFrame(pixels);
    }

    public void Finish()
    {
        ffmpeg.Finish();
    }
}