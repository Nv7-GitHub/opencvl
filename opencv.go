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

func (l *openCVLayer) execute(img image.Image) (image.Image, error) {
	mat, err := gocv.ImageToMatRGBA(img)
	if err != nil {
		return nil, err
	}
	mat = l.process(mat, l.args...)
	img, err = mat.ToImage()
	if err != nil {
		return nil, err
	}
	return img, nil
}

func (l *openCVLayer) setArgs(args []interface{}) {
	l.args = args
}

// NewOpenCVLayer creates a new opencv layer, given the process function
func NewOpenCVLayer(process func(mat gocv.Mat, args ...interface{}) gocv.Mat) Layer {
	return &openCVLayer{
		process: process,
		args:    make([]interface{}, 0),
	}
}
