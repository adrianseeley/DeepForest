public static class Synthetic
{
    public static List<Sample> DropoutFeatures(List<Sample> samples, float dropoutRate, int copies)
    {
        Random random = new Random();
        List<Sample> result = new List<Sample>();
        foreach(Sample sample in samples)
        {
            for (int copy = 0; copy < copies; copy++)
            {
                float[] input = new float[sample.input.Length];
                sample.input.CopyTo(input, 0);
                for (int i = 0; i < input.Length; i++)
                {
                    if (random.NextSingle() < dropoutRate)
                    {
                        input[i] = 0f;
                    }
                }
                result.Add(new Sample(input, sample.output));
            }
        }
        return result;
    }

}