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

    public static float[,] CreateDistanceMatrix(List<Sample> trainSamples, float[] inputWeights, float[] distanceWeights)
    {
        float[,] distanceMatrix = new float[trainSamples.Count, trainSamples.Count];
        for (int a = 0; a < trainSamples.Count; a++)
        {
            float[] inputA = trainSamples[a].input;
            float distanceWeightA = distanceWeights[a];
            distanceMatrix[a, a] = 0f;
            for (int b = a + 1; b < trainSamples.Count; b++)
            {
                float[] inputB = trainSamples[b].input;
                float distanceWeightB = distanceWeights[b];
                float distance = 0f;
                for (int i = 0; i < inputA.Length; i++)
                {
                    distance += MathF.Abs(inputA[i] - inputB[i]) * inputWeights[i];
                }
                distanceMatrix[a, b] = distance * distanceWeightA;
                distanceMatrix[b, a] = distance * distanceWeightB;
            }
        }
        return distanceMatrix;
    }

    public static float[] CreateDistanceWeights(List<Sample> trainSamples)
    {
        float[] distanceWeights = new float[trainSamples.Count];
        for (int i = 0; i < distanceWeights.Length; i++)
        {
            distanceWeights[i] = 1f;
        }
        return distanceWeights;
    }

    public static float[] CreateContributionWeights(List<Sample> trainSamples)
    {
        float[] contributionWeights = new float[trainSamples.Count];
        for (int i = 0; i < contributionWeights.Length; i++)
        {
            contributionWeights[i] = 1f;
        }
        return contributionWeights;
    }

    public static void ModifyInputWeight(List<Sample> trainSamples, float[] inputWeights, float[] distanceWeights, float[,] distanceMatrix, int inputWeightIndex, float newInputWeightValue)
    {
        float oldInputWeightValue = inputWeights[inputWeightIndex];
        Parallel.For(0, trainSamples.Count, a =>
        {
            float[] inputA = trainSamples[a].input;
            float distanceWeightA = distanceWeights[a];
            for (int b = 0; b < trainSamples.Count; b++)
            {
                if (a == b)
                {
                    continue;
                }
                float[] inputB = trainSamples[b].input;
                float distance = distanceMatrix[a, b];
                distance /= distanceWeightA;
                distance -= MathF.Abs(inputA[inputWeightIndex] - inputB[inputWeightIndex]) * oldInputWeightValue;
                distance += MathF.Abs(inputA[inputWeightIndex] - inputB[inputWeightIndex]) * newInputWeightValue;
                distance *= distanceWeightA;
                distanceMatrix[a, b] = distance;
            }
        });
        inputWeights[inputWeightIndex] = newInputWeightValue;
    }

    public static void ModifyDistanceWeight(List<Sample> trainSamples, float[] inputWeights, float[] distanceWeights, float[,] distanceMatrix, int distanceWeightIndex, float newDistanceWeightValue)
    {
        float oldDistanceWeightValue = distanceWeights[distanceWeightIndex];
        float[] inputA = trainSamples[distanceWeightIndex].input;
        Parallel.For(0, trainSamples.Count, b =>
        {
            if (distanceWeightIndex == b)
            {
                return;
            }
            float[] inputB = trainSamples[b].input;
            float distance = distanceMatrix[distanceWeightIndex, b];
            distance /= oldDistanceWeightValue;
            distance *= newDistanceWeightValue;
            distanceMatrix[distanceWeightIndex, b] = distance;
        });
        distanceWeights[distanceWeightIndex] = newDistanceWeightValue;
    }

    public static void ModifyContributionWeight(float[] contributionWeights, int contributionWeightIndex, float newContributionWeightValue)
    {
        contributionWeights[contributionWeightIndex] = newContributionWeightValue;
    }

    public static List<(Sample trainSample, float distance, float contributionWeight)> FindNeighbours(int trainIndex, List<Sample> trainSamples, float[,] distanceMatrix, float[] contributionWeights)
    {
        List<(Sample trainSample, float distance, float contribution)> neighbours = new List<(Sample, float, float)>(trainSamples.Count - 1);
        for (int i = 0; i < trainSamples.Count; i++)
        {
            if (i == trainIndex)
            {
                continue;
            }
            float distance = distanceMatrix[i, trainIndex];
            float contributionWeight = contributionWeights[i];
            neighbours.Add((trainSamples[i], distance, contributionWeight));
        }
        neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
        return neighbours;
    }

    public static List<(Sample trainSample, float distance, float contribution)> FindNeighboursTest(List<Sample> trainSamples, Sample testSample, float[] inputWeights, float[] distanceWeights, float[] contributionWeights)
    {
        List<(Sample trainSample, float distance, float contribution)> neighbours = new List<(Sample, float, float)>(trainSamples.Count);
        for (int i = 0; i < trainSamples.Count; i++)
        {
            Sample trainSample = trainSamples[i];
            float distanceWeight = distanceWeights[i];
            float contributionWeight = contributionWeights[i];
            float distance = 0f;
            for (int j = 0; j < trainSample.input.Length; j++)
            {
                distance += MathF.Abs(trainSample.input[j] - testSample.input[j]) * inputWeights[j];
            }
            distance *= distanceWeight;
            neighbours.Add((trainSample, distance, contributionWeight));
        }
        neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
        return neighbours;
    }

    public static List<(int k, float[] prediction)> Predict(int kMinInclusive, int kMaxInclusive, List<(Sample trainSample, float distance, float contributionWeight)> neighbours)
    {
        int predictionSize = neighbours[0].trainSample.output.Length;
        List<(int k, float[] prediction)> predictions = new List<(int, float[])>(kMaxInclusive - kMinInclusive + 1);
        float weightSum = 0f;
        float[] predictionSum = new float[predictionSize];
        for (int k = 0; k < kMaxInclusive; k++)
        {
            Sample neighbour = neighbours[k].trainSample;
            float distance = neighbours[k].distance;
            float contribution = neighbours[k].contributionWeight;
            float weight = contribution / (distance + 0.0000001f);
            weightSum += weight;
            for (int j = 0; j < predictionSize; j++)
            {
                predictionSum[j] += neighbour.output[j] * weight;
            }
            if (k + 1 >= kMinInclusive)
            {
                float[] prediction = new float[predictionSize];
                for (int j = 0; j < predictionSize; j++)
                {
                    prediction[j] = predictionSum[j] / weightSum;
                }
                predictions.Add((k + 1, prediction));
            }
        }
        return predictions;
    }

    public static float[] PredictTest(int k, List<(Sample trainSample, float distance, float contributionWeight)> neighbours)
    {
        int predictionSize = neighbours[0].trainSample.output.Length;
        float weightSum = 0f;
        float[] predictionSum = new float[predictionSize];
        for (int i = 0; i < k; i++)
        {
            Sample neighbour = neighbours[i].trainSample;
            float distance = neighbours[i].distance;
            float contribution = neighbours[i].contributionWeight;
            float weight = contribution / (distance + 0.0000001f);
            weightSum += weight;
            for (int j = 0; j < predictionSize; j++)
            {
                predictionSum[j] += neighbour.output[j] * weight;
            }
        }
        float[] prediction = new float[predictionSize];
        for (int j = 0; j < predictionSize; j++)
        {
            prediction[j] = predictionSum[j] / weightSum;
        }
        return prediction;
    }

    public static (int k, float distanceSum) DetermineBestK(List<(int k, float[] prediction)>[] kPredictions, List<Sample> trainSamples)
    {
        int bestK = -1;
        float bestDistanceSum = float.PositiveInfinity;
        object bestLock = new object();
        Parallel.For(0, kPredictions[0].Count, kIndex =>
        {
            int kValue = kPredictions[0][kIndex].k;
            float distanceSum = 0f;
            for (int i = 0; i < trainSamples.Count; i++)
            {
                float[] prediction = kPredictions[i][kIndex].prediction;
                float[] output = trainSamples[i].output;
                for (int j = 0; j < prediction.Length; j++)
                {
                    distanceSum += MathF.Abs(prediction[j] - output[j]);
                }
            }
            lock (bestLock)
            {
                if (distanceSum < bestDistanceSum)
                {
                    bestK = kValue;
                    bestDistanceSum = distanceSum;
                }
            }
        });
        return (bestK, bestDistanceSum);
    }

    public static (int k, float distanceSum) Evaluate(int kMinInclusive, int kMaxInclusive, List<Sample> trainSamples, float[,] distanceMatrix, float[] contributionWeights)
    {
        List<(int k, float[] prediction)>[] kPredictions = new List<(int k, float[] prediction)>[trainSamples.Count];
        Parallel.For(0, trainSamples.Count, i =>
        {
            List<(Sample trainSample, float distance, float contributionWeight)> neighbours = FindNeighbours(i, trainSamples, distanceMatrix, contributionWeights);
            List<(int k, float[] prediction)> predictions = Predict(kMinInclusive, kMaxInclusive, neighbours);
            kPredictions[i] = predictions;
        });
        return DetermineBestK(kPredictions, trainSamples);
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

    public static int ScoreTest(int k, List<Sample> trainSamples, List<Sample> testSamples, float[] inputWeights, float[] distanceWeights, float[] contributionWeights)
    {
        int correct = 0;
        Parallel.ForEach(testSamples, testSample =>
        {
            List<(Sample trainSample, float distance, float contribution)> neighbours = FindNeighboursTest(trainSamples, testSample, inputWeights, distanceWeights, contributionWeights);
            float[] prediction = PredictTest(k, neighbours);
            if (Argmax(prediction) == Argmax(testSample.output))
            {
                Interlocked.Increment(ref correct);
            }
        });
        return correct;
    }

    public enum NudgeType
    {
        InputWeight,
        DistanceWeight,
        ContributionWeight
    };

    public enum NudgeDirection
    {
        Increase,
        Decrease
    };

    public static void Log(TextWriter csv, int epoch, int fails, int k, float distanceSum, int testCorrect, float[] inputWeights, float[] distanceWeights, float[] contributionWeights)
    {
        float inputWeightSum = inputWeights.Sum();
        float distanceWeightSum = distanceWeights.Sum();
        float contributionWeightSum = contributionWeights.Sum();
        csv.Write($"{epoch},{fails},{k},{distanceSum},{testCorrect},{inputWeightSum},{distanceWeightSum},{contributionWeightSum}");
        foreach (float inputWeight in inputWeights)
        {
            csv.Write($",{inputWeight}");
        }
        foreach (float distanceWeight in distanceWeights)
        {
            csv.Write($",{distanceWeight}");
        }
        foreach (float contributionWeight in contributionWeights)
        {
            csv.Write($",{contributionWeight}");
        }
        csv.WriteLine();
        csv.Flush();
        Console.WriteLine($"epoch: {epoch}, fails: {fails}, k: {k}, distanceSum: {distanceSum}, testCorrect: {testCorrect}, inputWeightSum: {inputWeightSum}, distanceWeightSum: {distanceWeightSum}, contributionWeightSum: {contributionWeightSum}");
    }

    public static void Main()
    {
        Random random = new Random();
        float nudgeAmount = 0.01f;
        int kMinInclusive = 1;
        int kMaxInclusive = 10;
        int epoch = 0;
        int fails = 0;

        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> trainSamples, List<Sample> testSamples) = Sample.Conormalize(mnistTrain, mnistTest);

        float[] inputWeights = CreateInputWeights(trainSamples);
        float[] distanceWeights = CreateDistanceWeights(trainSamples);
        float[] contributionWeights = CreateContributionWeights(trainSamples);
        float[,] distanceMatrix = CreateDistanceMatrix(trainSamples, inputWeights, distanceWeights);

        List<(NudgeType nudgeType, NudgeDirection nudgeDirection, int index)> mutations = new List<(NudgeType nudgeType, NudgeDirection nudgeDirection, int index)>();
        for (int i = 0; i < inputWeights.Length; i++)
        {
            mutations.Add((NudgeType.InputWeight, NudgeDirection.Increase, i));
            mutations.Add((NudgeType.InputWeight, NudgeDirection.Decrease, i));
        }
        for (int i = 0; i < distanceWeights.Length; i++)
        {
            mutations.Add((NudgeType.DistanceWeight, NudgeDirection.Increase, i));
            mutations.Add((NudgeType.DistanceWeight, NudgeDirection.Decrease, i));
        }
        for (int i = 0; i < contributionWeights.Length; i++)
        {
            mutations.Add((NudgeType.ContributionWeight, NudgeDirection.Increase, i));
            mutations.Add((NudgeType.ContributionWeight, NudgeDirection.Decrease, i));
        }

        TextWriter csv = new StreamWriter("results.csv", append: false);
        csv.Write("epoch,fails,k,distanceSum,testCorrect,inputWeightSum,distanceWeightSum,contributionWeightSum");
        for (int i = 0; i < inputWeights.Length; i++)
        {
            csv.Write($",inputWeight{i}");
        }
        for (int i = 0; i < distanceWeights.Length; i++)
        {
            csv.Write($",distanceWeight{i}");
        }
        for (int i = 0; i < contributionWeights.Length; i++)
        {
            csv.Write($",contributionWeight{i}");
        }
        csv.WriteLine();
        csv.Flush();


        // calculate starting point
        (int k, float distanceSum) bestEvaluation = Evaluate(kMinInclusive, kMaxInclusive, trainSamples, distanceMatrix, contributionWeights);
        int bestScore = ScoreTest(bestEvaluation.k, trainSamples, testSamples, inputWeights, distanceWeights, contributionWeights);

        // log starting point
        Log(csv, epoch, fails, bestEvaluation.k, bestEvaluation.distanceSum, bestScore, inputWeights, distanceWeights, contributionWeights);

        // loop until improvements stop
        bool improved = true;
        while (improved)
        {
            improved = false;

            // shuffle mutations
            for (int i = 0; i < mutations.Count; i++)
            {
                int j = random.Next(i, mutations.Count);
                (mutations[i], mutations[j]) = (mutations[j], mutations[i]); 
            }

            // iterate shuffled mutations
            foreach ((NudgeType nudgeType, NudgeDirection nudgeDirection, int index) mutation in mutations)
            {
                // get the old inputWeight
                float oldWeight;
                switch (mutation.nudgeType)
                {
                    case NudgeType.InputWeight:
                        oldWeight = inputWeights[mutation.index];
                        break;
                    case NudgeType.DistanceWeight:
                        oldWeight = distanceWeights[mutation.index];
                        break;
                    case NudgeType.ContributionWeight:
                        oldWeight = contributionWeights[mutation.index];
                        break;
                    default:
                        throw new Exception("Invalid nudge type");
                }

                // determine the new inputWeight
                float newWeight;
                switch (mutation.nudgeDirection)
                {
                    case NudgeDirection.Increase:
                        newWeight = oldWeight + nudgeAmount;
                        break;
                    case NudgeDirection.Decrease:
                        if (oldWeight == 0f)
                        {
                            continue; // cant make a inputWeight negative, skip this mutation
                        }
                        newWeight = oldWeight - nudgeAmount;
                        if (newWeight < 0f)
                        {
                            newWeight = 0f;
                        }
                        break;
                    default:
                        throw new Exception("Invalid nudge direction");
                }

                // modify the inputWeight
                switch (mutation.nudgeType)
                {
                    case NudgeType.InputWeight:
                        ModifyInputWeight(trainSamples, inputWeights, distanceWeights, distanceMatrix, mutation.index, newWeight);
                        break;
                    case NudgeType.DistanceWeight:
                        ModifyDistanceWeight(trainSamples, inputWeights, distanceWeights, distanceMatrix, mutation.index, newWeight);
                        break;
                    case NudgeType.ContributionWeight:
                        ModifyContributionWeight(contributionWeights, mutation.index, newWeight);
                        break;
                    default:
                        throw new Exception("Invalid nudge type");
                }

                // evaluate the new weights
                (int k, float distanceSum) evaluation = Evaluate(kMinInclusive, kMaxInclusive, trainSamples, distanceMatrix, contributionWeights);

                // determine result of evaluation
                bool keepMutation = false;
                switch (mutation.nudgeDirection)
                {
                    case NudgeDirection.Increase:
                        // increasing weights must show improvement to be kept
                        if (evaluation.distanceSum < bestEvaluation.distanceSum)
                        {
                            keepMutation = true;
                        }
                        break;
                    case NudgeDirection.Decrease:
                        // decreasing weights can be kept if they don't make things worse
                        if (evaluation.distanceSum <= bestEvaluation.distanceSum)
                        {
                            keepMutation = true;
                        }
                        break;
                    default:
                        throw new Exception("Invalid nudge direction");
                }

                // if we are keeping the mutation
                if (keepMutation)
                {
                    // score and mark it
                    improved = true;
                    bestEvaluation = evaluation;
                    bestScore = ScoreTest(bestEvaluation.k, trainSamples, testSamples, inputWeights, distanceWeights, contributionWeights);

                    // log it
                    Log(csv, epoch, fails, bestEvaluation.k, bestEvaluation.distanceSum, bestScore, inputWeights, distanceWeights, contributionWeights);

                    // increment epoch
                    epoch++;
                    continue;
                }

                // otherwise we need to undo the mutation
                switch (mutation.nudgeType)
                {
                    case NudgeType.InputWeight:
                        ModifyInputWeight(trainSamples, inputWeights, distanceWeights, distanceMatrix, mutation.index, oldWeight);
                        break;
                    case NudgeType.DistanceWeight:
                        ModifyDistanceWeight(trainSamples, inputWeights, distanceWeights, distanceMatrix, mutation.index, oldWeight);
                        break;
                    case NudgeType.ContributionWeight:
                        ModifyContributionWeight(contributionWeights, mutation.index, oldWeight);
                        break;
                    default:
                        throw new Exception("Invalid nudge type");
                }

                // increment fails
                fails++;
            }
        }

    }
}