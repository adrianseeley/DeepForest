/// <summary>
/// Samples are assumed to be prenormalized in both the input and output dimensions.
/// </summary>

public class GraphDiffusion
{
    public Random random;
    public List<GraphDiffusionNode> nodes;

    public GraphDiffusion(List<Sample> samples, int k, int diffusionSteps, int transferRegions)
    {
        this.random = new Random();

        // convert all samples to nodes
        nodes = new List<GraphDiffusionNode>();
        foreach (Sample sample in samples)
        {
            nodes.Add(new GraphDiffusionNode(sample));
        }

        // add transfer regions
        for (int i = 0; i < transferRegions; i++)
        {
            Console.Write($"\rCreating Transfer Region: {i + 1}/{transferRegions}");
            float[] transferRegionInput = new float[samples[0].input.Length];
            for (int j = 0; j < transferRegionInput.Length; j++)
            {
                transferRegionInput[j] = random.NextSingle();
            }
            nodes.Add(new GraphDiffusionNode(transferRegionInput, nodes[0].output.Length));
        }
        Console.WriteLine();

        // calculate distances between all nodes (we skip a==b which defaults to 0 anyways)
        float[,] distances = new float[nodes.Count, nodes.Count];
        float distanceMin = float.PositiveInfinity;
        float distanceMax = float.NegativeInfinity;
        for (int a = 0; a < nodes.Count; a++)
        {
            Console.Write($"\rCalculating Distances: {a + 1}/{nodes.Count}");
            Parallel.For(a + 1, nodes.Count, b =>
            {
                float distance = EuclideanDistance(nodes[a].input, nodes[b].input);
                distances[a, b] = distance;
                distances[b, a] = distance;
                distanceMin = MathF.Min(distanceMin, distance);
                distanceMax = MathF.Max(distanceMax, distance);
            });
        }
        float distanceRange = distanceMax - distanceMin;

        Console.WriteLine();

        // calculate normalized distances between all nodes (again skipping a==b which defaults to 0)
        float[,] normalizedDistances = new float[nodes.Count, nodes.Count];
        for (int a = 0; a < nodes.Count; a++)
        {
            Console.Write($"\rNormalizing Distances: {a + 1}/{nodes.Count}");
            for (int b = a + 1; b < nodes.Count; b++)
            {
                float normalizedDistance = (distances[a, b] - distanceMin) / distanceRange;
                normalizedDistances[a, b] = normalizedDistance;
                normalizedDistances[b, a] = normalizedDistance;
            }
        }
        Console.WriteLine();

        // each node requires a list of neighbours to pull from, find those neighbours
        for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
        {
            Console.Write($"\rFinding Neighbours: {nodeIndex + 1}/{nodes.Count}");

            // get the node in question
            GraphDiffusionNode node = nodes[nodeIndex];
            
            // create a list of neighbour distances
            List<(GraphDiffusionNode otherNode, float distance)> neighbourDistances = new List<(GraphDiffusionNode otherNode, float distance)>();

            // iterate through all other nodes to find the neighbours
            for (int otherNodexIndex = 0; otherNodexIndex < nodes.Count; otherNodexIndex++)
            {
                // cant have self as a neighbour
                if (otherNodexIndex == nodeIndex)
                {
                    continue;
                }

                // add to list
                neighbourDistances.Add((nodes[otherNodexIndex], normalizedDistances[nodeIndex, otherNodexIndex]));
            }

            // sort by distance close to far
            neighbourDistances.Sort((a, b) => a.distance.CompareTo(b.distance));

            // take the k nearest neighbours
            node.neighbours = neighbourDistances.Take(k).Select(a => a.otherNode).ToList();
        }
        Console.WriteLine();

        // create targets
        List<float[]> nodeTargets = new List<float[]>();
        foreach (GraphDiffusionNode node in nodes)
        {
            nodeTargets.Add(new float[node.output.Length]);
        }

        // make diffusion steps
        for (int diffusionStep = 0; diffusionStep < diffusionSteps; diffusionStep++)
        {
            Console.Write($"\rDiffusing: {diffusionStep + 1}/{diffusionSteps}");

            // iterate through nodes to generate the targets
            for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
            {
                // get the node
                GraphDiffusionNode node = nodes[nodeIndex];

                // get the node target
                float[] nodeTarget = nodeTargets[nodeIndex];

                // clear it
                Array.Clear(nodeTarget, 0, nodeTarget.Length);

                // track the influence weight sum
                float influenceWeightSum = 0f;

                // if this node is a producer add its influence
                if (node.sample != null)
                {
                    for (int i = 0; i < nodeTarget.Length; i++)
                    {
                        nodeTarget[i] += node.sample.output[i];
                    }
                    influenceWeightSum += 1f;
                }

                // iterate through the neighbours to influence the target
                foreach(GraphDiffusionNode neighbour in node.neighbours)
                {
                    // add influence
                    for (int i = 0; i < nodeTarget.Length; i++)
                    {
                        nodeTarget[i] += neighbour.output[i];
                    }
                    influenceWeightSum += 1f;
                }

                // weighted average the target by the influence weight sum
                for (int i = 0; i < nodeTarget.Length; i++)
                {
                    nodeTarget[i] /= influenceWeightSum;
                }
            }

            // iterate through nodes to update the outputs based on targets
            for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
            {
                // get the node
                GraphDiffusionNode node = nodes[nodeIndex];

                // get the node target
                float[] nodeTarget = nodeTargets[nodeIndex];

                // update the node output (average it with the target)
                for (int i = 0; i < node.output.Length; i++)
                {
                    node.output[i] += nodeTarget[i];
                    node.output[i] /= 2f;
                }
            }
        }
        Console.WriteLine();
    }

    public float[] LookupKNN(float[] input, int k)
    {
        List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            float distance = EuclideanDistance(input, node.input);
            nodeDistances.Add((node, distance));
        }
        nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int minK = Math.Min(k, nodeDistances.Count);
        float[] output = new float[nodes[0].output.Length];
        for (int i = 0; i < minK; i++)
        {
            GraphDiffusionNode node = nodeDistances[i].node;
            if (node.sample == null)
            {
                throw new Exception("Unreachable.");
            }
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += node.sample.output[j];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= minK;
        }
        return output;
    }

    public float[] LookupKNNDiffuse(float[] input, int k)
    {
        List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
        foreach (GraphDiffusionNode node in nodes)
        {
            float distance = EuclideanDistance(input, node.input);
            nodeDistances.Add((node, distance));
        }
        nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int minK = Math.Min(k, nodeDistances.Count);
        float[] output = new float[nodes[0].output.Length];
        for (int i = 0; i < minK; i++)
        {
            GraphDiffusionNode node = nodeDistances[i].node;
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += node.output[j];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= minK;
        }
        return output;
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
}

public class GraphDiffusionNode
{
    public float[] input;
    public float[] output;
    public Sample? sample;
    public List<GraphDiffusionNode> neighbours;

    public GraphDiffusionNode(float[] input, int outputSize)
    {
        this.input = input;
        this.output = new float[outputSize];
        this.sample = null;
        this.neighbours = new List<GraphDiffusionNode>();
    }

    public GraphDiffusionNode(Sample sample)
    {
        this.input = new float[sample.input.Length];
        sample.input.CopyTo(this.input, 0);
        this.output = new float[sample.output.Length];
        this.sample = sample;
        this.neighbours = new List<GraphDiffusionNode>();
    }
}