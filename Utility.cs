﻿public static class Utility
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

    public static float MeanAbsoluteDistance(float[] a, float[] b)
    {
        float distance = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            distance += MathF.Abs(a[i] - b[i]);
        }
        return distance / (float)a.Length;
    }

    public static float MeanAbsoluteDistance(float[] a, float[] b, List<int> features)
    {
        float distance = 0f;
        for (int i = 0; i < features.Count; i++)
        {
            distance += MathF.Abs(a[features[i]] - b[features[i]]);
        }
        return distance / (float)features.Count;
    }

    public static float EuclideanDistance(float[] a, float[] b)
    {
        float distance = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            distance += MathF.Pow(a[i] - b[i], 2);
        }
        return MathF.Sqrt(distance);
    }

    public static float EuclideanDistance(float[] a, float[] b, List<int> features)
    {
        float distance = 0f;
        for (int i = 0; i < features.Count; i++)
        {
            distance += MathF.Pow(a[features[i]] - b[features[i]], 2);
        }
        return MathF.Sqrt(distance);
    }
}