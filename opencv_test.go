package opencvl

import (
	"image"
	"image/png"
	"os"
	"testing"

	"gocv.io/x/gocv"
)

func TestOpenCVLayer(t *testing.T) {
	pipeline := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.BitwiseNot(mat, &mat)
		return mat
	})
	err := pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	file, err := os.Open("gopher.png")
	if err != nil {
		panic(err)
	}
	img, err := png.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

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

func TestOpenCVLayer_Video(t *testing.T) {
	pipeline := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.BitwiseNot(mat, &mat)
		return mat
	})
	err := pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	err = pipeline.ExecuteOnVideo("sample.mp4", "out.mp4")
	if err != nil {
		panic(err)
	}
}

func BenchmarkOpenCVLayer(b *testing.B) {
	pipeline := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.BitwiseNot(mat, &mat)
		return mat
	})
	err := pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	file, err := os.Open("gopher.png")
	if err != nil {
		panic(err)
	}
	img, err := png.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	for i := 0; i < b.N; i++ {
		img, err = pipeline.ExecuteOnImage(img.(*image.RGBA))
		if err != nil {
			panic(err)
		}
	}
}
