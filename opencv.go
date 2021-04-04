package opencvl

import (
	"image"

	"gocv.io/x/gocv"
)

type openCVLayer struct {
	process func(mat gocv.Mat, args ...interface{}) gocv.Mat
	args    []interface{}
}

func (l *openCVLayer) Type() string {
	return "opencv"
}

func (l *openCVLayer) execute(img *image.RGBA, time int) (*image.RGBA, error) {
	mat, err := gocv.ImageToMatRGB(img)
	if err != nil {
		return nil, err
	}
	mat = l.process(mat, append([]interface{}{time}, l.args...)...)
	out, err := mat.ToImage()
	if err != nil {
		return nil, err
	}
	return out.(*image.RGBA), nil
}

func (l *openCVLayer) setArgs(args []interface{}) error {
	l.args = args
	return nil
}

func (l *openCVLayer) build() error { return nil }

func (l *openCVLayer) cleanup() {}

// NewOpenCVLayer creates a new opencv layer, given the process function
func NewOpenCVLayer(process func(mat gocv.Mat, args ...interface{}) gocv.Mat) Layer {
	return &openCVLayer{
		process: process,
		args:    make([]interface{}, 0),
	}
}
