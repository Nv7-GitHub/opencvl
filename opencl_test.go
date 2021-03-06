package opencvl

import (
	"image"
	"image/png"
	"io/ioutil"
	"os"
	"testing"

	"gocv.io/x/gocv"
)

func TestOpenCLLayer(t *testing.T) {
	file, err := os.Open("gopher.png")
	if err != nil {
		panic(err)
	}
	img, err := png.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	var kernelSource = `
	__kernel void invert(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int time)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		const int2 dim = get_image_dim(in);
		float4 pixel = (float4)(0);
		if (pos.x < dim.x && pos.y < dim.y) {
			pixel = read_imagef(in, pos);
			pixel = (float4)(1) - pixel;
			pixel.w = 1;
			write_imagef(out, pos, pixel);
		}
	}`

	pipeline := NewPipeline()
	layer, err := NewOpenCLLayer(kernelSource, "invert", 0, 0)
	if err != nil {
		panic(err)
	}
	err = pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	err = pipeline.Build()
	if err != nil {
		panic(err)
	}

	img, err = pipeline.ExecuteOnImage(img.(*image.RGBA))
	if err != nil {
		panic(err)
	}

	file, err = os.OpenFile("out.png", os.O_WRONLY|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	err = png.Encode(file, img)
	if err != nil {
		panic(err)
	}
}

func TestOpenCLLayer_Video(t *testing.T) {
	var kernelSource = `
	__kernel void invert(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int time)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		const int2 dim = get_image_dim(in);
		float4 pixel = (float4)(0);
		if (pos.x < dim.x && pos.y < dim.y) {
			pixel = read_imagef(in, pos);
			pixel = (float4)(1) - pixel;
			pixel.w = 1;
			write_imagef(out, pos, pixel);
		}
	}`

	pipeline := NewPipeline()
	layer, err := NewOpenCLLayer(kernelSource, "invert", 0, 0)
	if err != nil {
		panic(err)
	}
	err = pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	err = pipeline.Build()
	if err != nil {
		panic(err)
	}

	err = pipeline.ExecuteOnVideo("sample.mp4", "outcl.mp4")
	if err != nil {
		panic(err)
	}
}

func BenchmarkOpenCLLayer(b *testing.B) {
	file, err := os.Open("gopher.png")
	if err != nil {
		panic(err)
	}
	img, err := png.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	var kernelSource = `
	__kernel void invert(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int time)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		const int2 dim = get_image_dim(in);
		float4 pixel = (float4)(0);
		if (pos.x < dim.x && pos.y < dim.y) {
			pixel = read_imagef(in, pos);
			pixel = (float4)(1) - pixel;
			pixel.w = 1;
			write_imagef(out, pos, pixel);
		}
	}`

	pipeline := NewPipeline()
	layer, err := NewOpenCLLayer(kernelSource, "invert", 0, 0)
	if err != nil {
		panic(err)
	}
	err = pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	err = pipeline.Build()
	if err != nil {
		panic(err)
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		img, err = pipeline.ExecuteOnImage(img.(*image.RGBA))
		if err != nil {
			panic(err)
		}
	}
}

func TestOpenCLLayer_Mandelbrot(t *testing.T) {
	source, err := ioutil.ReadFile("mandelbrot.cl")
	if err != nil {
		panic(err)
	}

	pipeline := NewPipeline()
	layer, err := NewOpenCLLayer(string(source), "mandelbrot", 0, 0)
	if err != nil {
		panic(err)
	}
	err = pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	hsv2rgb := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.CvtColor(mat, &mat, gocv.ColorHSVToBGR)
		return mat
	})
	pipeline.AddLayer(hsv2rgb)

	err = pipeline.Build()
	if err != nil {
		panic(err)
	}

	err = pipeline.SetOpenCLArgs(1)
	if err != nil {
		panic(err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 1920, 1080))

	img, err = pipeline.ExecuteOnImage(img)
	if err != nil {
		panic(err)
	}

	file, err := os.OpenFile("out.png", os.O_WRONLY|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	err = png.Encode(file, img)
	if err != nil {
		panic(err)
	}
}

func TestOpenCLLayer_MandelbrotZoom(t *testing.T) {
	source, err := ioutil.ReadFile("mandelbrot.cl")
	if err != nil {
		panic(err)
	}

	pipeline := NewPipeline()
	layer, err := NewOpenCLLayer(string(source), "mandelbrot", 0, 0)
	if err != nil {
		panic(err)
	}
	err = pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	hsv2rgb := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.CvtColor(mat, &mat, gocv.ColorHSVToBGR)
		return mat
	})
	pipeline.AddLayer(hsv2rgb)

	err = pipeline.Build()
	if err != nil {
		panic(err)
	}

	err = pipeline.SetOpenCLArgs(374)
	if err != nil {
		panic(err)
	}

	err = pipeline.ExecuteOnVideo("blank.mp4", "zoom.mp4")
	if err != nil {
		panic(err)
	}
}
