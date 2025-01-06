public static class Bitmap
{
    public static byte[,,] Create(int width, int height)
    {
        byte[,,] pixels = new byte[height, width, 3];
        return pixels;
    }

    public static void Clear(byte[,,] pixels, byte r, byte g, byte b)
    {
        int height = pixels.GetLength(0);
        int width = pixels.GetLength(1);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                pixels[y, x, 0] = r;
                pixels[y, x, 1] = g;
                pixels[y, x, 2] = b;
            }
        }
    }

    public static void DrawPixel(byte[,,] pixels, int x, int y, byte r, byte g, byte b)
    {
        int height = pixels.GetLength(0);
        int width = pixels.GetLength(1);
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return;
        }
        pixels[y, x, 0] = r;
        pixels[y, x, 1] = g;
        pixels[y, x, 2] = b;
    }

    public static void DrawLine(byte[,,] pixels, int x0, int y0, int x1, int y1, byte r, byte g, byte b)
    {
        int dx = Math.Abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -Math.Abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        int x = x0, y = y0;
        while (true)
        {
            DrawPixel(pixels, x, y, r, g, b);
            if (x == x1 && y == y1)
            {
                break;
            }
            int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y += sy;
            }
        }
    }

    public static void DrawCross(byte[,,] pixels, int cx, int cy, int wingSize, byte r, byte g, byte b)
    {
        DrawLine(pixels, cx - wingSize, cy, cx + wingSize, cy, r, g, b);
        DrawLine(pixels, cx, cy - wingSize, cx, cy + wingSize, r, g, b);
    }

    public static void DrawRect(byte[,,] pixels, int x, int y, int w, int h, byte r, byte g, byte b)
    {
        DrawLine(pixels, x, y, x + w, y, r, g, b);
        DrawLine(pixels, x, y + h, x + w, y + h, r, g, b);
        DrawLine(pixels, x, y, x, y + h, r, g, b);
        DrawLine(pixels, x + w, y, x + w, y + h, r, g, b);
    }

    public static void DrawCircle(byte[,,] pixels, int cx, int cy, int radius, byte r, byte g, byte b)
    {
        int x = radius;
        int y = 0;
        int err = 0;
        while (x >= y)
        {
            DrawPixel(pixels, cx + x, cy + y, r, g, b);
            DrawPixel(pixels, cx + y, cy + x, r, g, b);
            DrawPixel(pixels, cx - y, cy + x, r, g, b);
            DrawPixel(pixels, cx - x, cy + y, r, g, b);
            DrawPixel(pixels, cx - x, cy - y, r, g, b);
            DrawPixel(pixels, cx - y, cy - x, r, g, b);
            DrawPixel(pixels, cx + y, cy - x, r, g, b);
            DrawPixel(pixels, cx + x, cy - y, r, g, b);
            if (err <= 0)
            {
                y += 1;
                err += 2 * y + 1;
            }
            if (err > 0)
            {
                x -= 1;
                err -= 2 * x + 1;
            }
        }
    }

    public static void FillRect(byte[,,] pixels, int x, int y, int w, int h, byte r, byte g, byte b)
    {
        for (int j = y; j < y + h; j++)
        {
            for (int i = x; i < x + w; i++)
            {
                DrawPixel(pixels, i, j, r, g, b);
            }
        }
    }

    public static void FillCircle(byte[,,] pixels, int cx, int cy, int radius, byte r, byte g, byte b)
    {
        int x = radius;
        int y = 0;
        int err = 0;
        while (x >= y)
        {
            DrawLine(pixels, cx - x, cy + y, cx + x, cy + y, r, g, b);
            DrawLine(pixels, cx - y, cy + x, cx + y, cy + x, r, g, b);
            DrawLine(pixels, cx - x, cy - y, cx + x, cy - y, r, g, b);
            DrawLine(pixels, cx - y, cy - x, cx + y, cy - x, r, g, b);
            if (err <= 0)
            {
                y += 1;
                err += 2 * y + 1;
            }
            if (err > 0)
            {
                x -= 1;
                err -= 2 * x + 1;
            }
        }
    }

    public static void DrawArgmaxActionIndicator(byte[,,] pixels, int actionIndex, int actionCount)
    {
        int insetX = 10;
        int insetY = 10;
        int segmentWidth = 40;
        int segmentHeight = 5;
        int space = 3;
        for (int i = 0; i < actionCount; i++)
        {
            int x = insetX;
            int y = insetY + i * (segmentHeight + space);
            if (i == actionIndex)
            {
                FillRect(pixels, x, y, segmentWidth, segmentHeight, 0, 255, 0);
            }
            else
            {
                FillRect(pixels, x, y, segmentWidth, segmentHeight, 255, 0, 0);
            }
            DrawRect(pixels, x, y, segmentWidth, segmentHeight, 0, 0, 0);
        }
    }

    public static void DrawContinuousActionIndicators(byte[,,] pixels, float[] action, float[] actionMins, float[] actionMaxs)
    {
        int insetX = 10;
        int insetY = 10;
        int sideWidth = 20;
        int segmentHeight = 5;
        int space = 3;

        for (int i = 0; i < action.Length; i++)
        {
            float normal = (action[i] - actionMins[i]) / (actionMaxs[i] - actionMins[i]);
            int x = insetX;
            int y = insetY + i * (segmentHeight + space);
            int width = (int)(normal * sideWidth);
            if (action[i] < 0)
            {
                FillRect(pixels, x + sideWidth - width, y, width, segmentHeight, 255, 0, 0);
            }
            else
            {
                FillRect(pixels, x + sideWidth, y, width, segmentHeight, 0, 255, 0);
            }
            DrawRect(pixels, x, y, sideWidth, segmentHeight, 0, 0, 0);
            DrawRect(pixels, x + sideWidth, y, sideWidth, segmentHeight, 0, 0, 0);
        }
    }


    public static void SaveToBMP(string filename, byte[,,] pixels)
    {
        int height = pixels.GetLength(0);
        int width = pixels.GetLength(1);
        int rowSize = (width * 3 + 3) & ~3;
        int pixelDataSize = rowSize * height;
        int fileSize = 14 + 40 + pixelDataSize;
        byte[] fileHeader = new byte[14];
        fileHeader[0] = (byte)'B';
        fileHeader[1] = (byte)'M';
        BitConverter.GetBytes(fileSize).CopyTo(fileHeader, 2);
        BitConverter.GetBytes(54).CopyTo(fileHeader, 10);
        byte[] dibHeader = new byte[40];
        BitConverter.GetBytes(40).CopyTo(dibHeader, 0);
        BitConverter.GetBytes(width).CopyTo(dibHeader, 4);
        BitConverter.GetBytes(height).CopyTo(dibHeader, 8);
        BitConverter.GetBytes((short)1).CopyTo(dibHeader, 12);
        BitConverter.GetBytes((short)24).CopyTo(dibHeader, 14);
        BitConverter.GetBytes(pixelDataSize).CopyTo(dibHeader, 20);
        using (var fs = new FileStream(filename, FileMode.Create, FileAccess.Write))
        {
            fs.Write(fileHeader, 0, fileHeader.Length);
            fs.Write(dibHeader, 0, dibHeader.Length);
            byte[] row = new byte[rowSize];
            for (int y = height - 1; y >= 0; y--)
            {
                int index = 0;
                for (int x = 0; x < width; x++)
                {
                    byte r = pixels[y, x, 0];
                    byte g = pixels[y, x, 1];
                    byte b = pixels[y, x, 2];
                    row[index++] = b;
                    row[index++] = g;
                    row[index++] = r;
                }
                while (index < rowSize)
                {
                    row[index++] = 0;
                }
                fs.Write(row, 0, rowSize);
            }
        }
    }
}