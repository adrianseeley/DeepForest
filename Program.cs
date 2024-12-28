public class Program
{
    public static void Main()
    {
        List<(List<float> input, List<float> output)> samples = new List<(List<float> input, List<float> output)>();
        samples.Add((new List<float> { 0, 0 }, new List<float> { 0 }));
        samples.Add((new List<float> { 1, 0 }, new List<float> { 1 }));
        samples.Add((new List<float> { 0, 1 }, new List<float> { 1 }));
        samples.Add((new List<float> { 1, 1 }, new List<float> { 0 }));
        MultiDeepRandomForest multiDeepRandomForest = new MultiDeepRandomForest(samples, 10, 10);

        foreach ((List<float> input, List<float> output) sample in samples)
        {
            List<float> predictOutput = multiDeepRandomForest.Predict(sample.input);
            Console.WriteLine("Actual: " + string.Join(", ", sample.output) + " Predict: " + string.Join(", ", predictOutput));
        }
    }
}