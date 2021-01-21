package opencvl

import (
	"image"
	"image/png"
	"os"
	"testing"
)

func TestBuiltins(t *testing.T) {
	p := RotatePipeline(480/2, 270/2, 75)

	file, err := os.Open("gopher.png")
	if err != nil {
		panic(err)
	}
	img, err := png.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	img, err = p.ExecuteOnImage(img.(*image.RGBA))
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

	err = p.ExecuteOnVideo("sample.mp4", "out.mp4")
	if err != nil {
		panic(err)
	}
}
