// +build videos

package opencvl

import (
	"io/ioutil"
	"testing"

	"gocv.io/x/gocv"
)

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

	err = pipeline.SetOpenCLArgs(498)
	if err != nil {
		panic(err)
	}

	err = pipeline.ExecuteOnVideo("blank.mp4", "mandelzoom.mp4")
	if err != nil {
		panic(err)
	}
}

func TestOpenCLLayer_BurningShipZoom(t *testing.T) {
	source, err := ioutil.ReadFile("mandelbrot.cl")
	if err != nil {
		panic(err)
	}

	pipeline := NewPipeline()
	layer, err := NewOpenCLLayer(string(source), "burning_ship", 0, 0)
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

	err = pipeline.SetOpenCLArgs(498)
	if err != nil {
		panic(err)
	}

	err = pipeline.ExecuteOnVideo("blank.mp4", "burningshipzoom.mp4")
	if err != nil {
		panic(err)
	}
}
