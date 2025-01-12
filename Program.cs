public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
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

    public static float[] CreateInputWeights(List<Sample> trainSamples)
    {
        float[] inputWeights = new float[trainSamples[0].input.Length];
        for (int i = 0; i < inputWeights.Length; i++)
        {
            inputWeights[i] = 1f;
        }
        return inputWeights;
    }

    public static float[,] CreateDistanceMatrix(List<Sample> trainSamples, float[] inputWeights)
    {
        float[,] distanceMatrix = new float[trainSamples.Count, trainSamples.Count];
        int complete = 0;
        Parallel.For(0, trainSamples.Count, a =>
        {
            float[] inputA = trainSamples[a].input;
            distanceMatrix[a, a] = 0f;
            for (int b = a + 1; b < trainSamples.Count; b++)
            {
                float[] inputB = trainSamples[b].input;
                float distance = 0f;
                for (int i = 0; i < inputA.Length; i++)
                {
                    distance += MathF.Abs(inputA[i] - inputB[i]) * inputWeights[i];
                }
                distanceMatrix[a, b] = distance;
                distanceMatrix[b, a] = distance;
            }
            int localComplete = Interlocked.Increment(ref complete);
            if (localComplete % 1000 == 0)
            {
                Console.WriteLine($"Matrix: {localComplete} / {trainSamples.Count}");
            }
        });
        return distanceMatrix;
    }

    public static void ModifyInputWeight(List<Sample> trainSamples, float[] inputWeights, float[,] distanceMatrix, int inputWeightIndex, float newInputWeightValue)
    {
        float oldInputWeightValue = inputWeights[inputWeightIndex];
        Parallel.For(0, trainSamples.Count, a =>
        {
            float[] inputA = trainSamples[a].input;
            for (int b = a + 1; b < trainSamples.Count; b++)
            {
                float[] inputB = trainSamples[b].input;
                float distance = distanceMatrix[a, b];
                distance -= MathF.Abs(inputA[inputWeightIndex] - inputB[inputWeightIndex]) * oldInputWeightValue;
                distance += MathF.Abs(inputA[inputWeightIndex] - inputB[inputWeightIndex]) * newInputWeightValue;
                distanceMatrix[a, b] = distance;
                distanceMatrix[b, a] = distance;
            }
        });
        inputWeights[inputWeightIndex] = newInputWeightValue;
    }

    public static List<(Sample trainSample, float distance)> FindNeighbours(int k, int trainIndex, List<Sample> trainSamples, float[,] distanceMatrix)
    {
        List<(Sample trainSample, float distance)> neighbours = new List<(Sample, float)>(k);
        for (int i = 0; i < trainSamples.Count; i++)
        {
            if (i == trainIndex)
            {
                continue;
            }
            float distance = distanceMatrix[i, trainIndex];
            if (neighbours.Count < k)
            {
                neighbours.Add((trainSamples[i], distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                continue;
            }
            if (distance < neighbours.Last().distance)
            {
                neighbours.Add((trainSamples[i], distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                neighbours.RemoveAt(neighbours.Count - 1);
                continue;
            }
        }
        return neighbours;
    }

    public static List<(Sample trainSample, float distance)> FindNeighboursTest(int k, List<Sample> trainSamples, Sample testSample, float[] inputWeights)
    {
        List<(Sample trainSample, float distance)> neighbours = new List<(Sample, float)>(k);
        for (int i = 0; i < trainSamples.Count; i++)
        {
            Sample trainSample = trainSamples[i];
            float distance = 0f;
            for (int j = 0; j < trainSample.input.Length; j++)
            {
                distance += MathF.Abs(trainSample.input[j] - testSample.input[j]) * inputWeights[j];
            }
            if (neighbours.Count < k)
            {
                neighbours.Add((trainSample, distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                continue;
            }
            if (distance < neighbours.Last().distance)
            {
                neighbours.Add((trainSample, distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                neighbours.RemoveAt(neighbours.Count - 1);
                continue;
            }
        }
        return neighbours;
    }

    public static float[] Predict(int k, List<(Sample trainSample, float distance)> neighbours)
    {
        int predictionSize = neighbours[0].trainSample.output.Length;
        float weightSum = 0f;
        float[] prediction = new float[predictionSize];
        for (int neighbourIndex = 0; neighbourIndex < neighbours.Count; neighbourIndex++)
        {
            Sample neighbour = neighbours[neighbourIndex].trainSample;
            float distance = neighbours[neighbourIndex].distance;
            float weight = 1f / (distance + 0.0000001f);
            weightSum += weight;
            for (int j = 0; j < predictionSize; j++)
            {
                prediction[j] += neighbour.output[j] * weight;
            }
        }
        for (int j = 0; j < predictionSize; j++)
        {
            prediction[j] /= weightSum;
        }
        return prediction;
    }

    public static float[] PredictTest(int k, List<(Sample trainSample, float distance)> neighbours)
    {
        int predictionSize = neighbours[0].trainSample.output.Length;
        float weightSum = 0f;
        float[] prediction = new float[predictionSize];
        for (int neighbourIndex = 0; neighbourIndex < neighbours.Count; neighbourIndex++)
        {
            Sample neighbour = neighbours[neighbourIndex].trainSample;
            float distance = neighbours[neighbourIndex].distance;
            float weight = 1f / (distance + 0.0000001f);
            weightSum += weight;
            for (int j = 0; j < predictionSize; j++)
            {
                prediction[j] += neighbour.output[j] * weight;
            }
        }
        for (int j = 0; j < predictionSize; j++)
        {
            prediction[j] /= weightSum;
        }
        return prediction;
    }

    public static float Evaluate(int k, List<Sample> trainSamples, float[,] distanceMatrix)
    {
        float[] localDistanceSums = new float[trainSamples.Count];
        Parallel.For(0, trainSamples.Count, trainIndex =>
        {
            float[] output = trainSamples[trainIndex].output;
            List<(Sample trainSample, float distance)> neighbours = FindNeighbours(k, trainIndex, trainSamples, distanceMatrix);
            float[] prediction = Predict(k, neighbours);
            float localDistanceSum = 0f;
            for (int j = 0; j < prediction.Length; j++)
            {
                localDistanceSum += MathF.Abs(prediction[j] - output[j]);
            }
            localDistanceSums[trainIndex] = localDistanceSum;
        });
        float distanceSum = 0f;
        for (int i = 0; i < trainSamples.Count; i++)
        {
            distanceSum += localDistanceSums[i];
        }
        return distanceSum;
    }

    public static int Argmax(float[] vector)
    {
        int index = 0;
        float value = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (vector[i] > value)
            {
                index = i;
                value = vector[i];
            }
        }
        return index;
    }

    public static int ScoreTest(int k, List<Sample> trainSamples, List<Sample> testSamples, float[] inputWeights)
    {
        int correct = 0;
        Parallel.ForEach(testSamples, testSample =>
        {
            List<(Sample trainSample, float distance)> neighbours = FindNeighboursTest(k, trainSamples, testSample, inputWeights);
            float[] prediction = PredictTest(k, neighbours);
            if (Argmax(prediction) == Argmax(testSample.output))
            {
                Interlocked.Increment(ref correct);
            }
        });
        return correct;
    }

    public static void Log(TextWriter csv, int epoch, int fails, float distanceSum, int testCorrect, float[] inputWeights)
    {
        float inputWeightSum = inputWeights.Sum();
        csv.WriteLine($"{epoch},{fails},{distanceSum},{testCorrect},{inputWeightSum}");
        csv.Flush();
        Console.WriteLine($"epoch: {epoch}, fails: {fails}, distanceSum: {distanceSum}, testCorrect: {testCorrect}, inputWeightSum: {inputWeightSum}");
    }

    public static void Main()
    {
        Random random = new Random();
        float nudgeAmount = 0.01f;
        int k = 3;
        int epoch = 0;
        int fails = 0;

        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> trainSamples, List<Sample> testSamples) = Sample.Conormalize(mnistTrain, mnistTest);

        float[] inputWeights = CreateInputWeights(trainSamples);
        float[,] distanceMatrix = CreateDistanceMatrix(trainSamples, inputWeights);
        List<int> mutationIndices = Enumerable.Range(0, inputWeights.Length).ToList();

        TextWriter csv = new StreamWriter("results.csv", append: false);
        csv.WriteLine("epoch,fails,distanceSum,testCorrect,inputWeightSum");
        csv.Flush();

        // calculate starting point
        float bestDistanceSum = Evaluate(k, trainSamples, distanceMatrix);
        int bestScore = ScoreTest(k, trainSamples, testSamples, inputWeights);

        // log starting point
        Log(csv, epoch, fails, bestDistanceSum, bestScore, inputWeights);

        // loop until improvements stop
        bool improved = true;
        while (improved)
        {
            improved = false;

            // shuffle mutations
            for (int i = 0; i < mutationIndices.Count; i++)
            {
                int j = random.Next(i, mutationIndices.Count);
                (mutationIndices[i], mutationIndices[j]) = (mutationIndices[j], mutationIndices[i]); 
            }

            // iterate shuffled mutations
            foreach (int mutationIndex in mutationIndices)
            {
                // get the old inputWeight
                float oldWeight = inputWeights[mutationIndex];

                // if its already 0 we cant reduce it more
                if (oldWeight == 0f)
                {
                    continue;
                }

                // determine the new inputWeight
                float newWeight = oldWeight - nudgeAmount;
                if (newWeight < 0f)
                {
                    newWeight = 0f;
                }

                // modify the inputWeight
                ModifyInputWeight(trainSamples, inputWeights, distanceMatrix, mutationIndex, newWeight);

                // evaluate the new weights
                float distanceSum = Evaluate(k, trainSamples, distanceMatrix);

                // if we are keeping the mutationIndex
                if (distanceSum <= bestDistanceSum)
                {
                    // increment epoch
                    epoch++;

                    // score and mark it
                    improved = true;
                    bestDistanceSum = distanceSum;
                    bestScore = ScoreTest(k, trainSamples, testSamples, inputWeights);

                    // log it
                    Log(csv, epoch, fails, bestDistanceSum, bestScore, inputWeights);
                    continue;
                }

                // otherwise we need to undo the mutationIndex
                ModifyInputWeight(trainSamples, inputWeights, distanceMatrix, mutationIndex, oldWeight);

                // increment fails
                fails++;
            }
        }

    }
}