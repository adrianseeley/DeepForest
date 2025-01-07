public static class Utility
{
    public static float[] Average(List<float[]> vectors)
    {
        float[] average = new float[vectors[0].Length];
        foreach (float[] vector in vectors)
        {
            for (int i = 0; i < average.Length; i++)
            {
                average[i] += vector[i];
            }
        }
        for (int i = 0; i < average.Length; i++)
        {
            average[i] /= vectors.Count;
        }
        return average;
    }
}