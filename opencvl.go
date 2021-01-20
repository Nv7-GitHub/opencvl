package opencvl

import "image"

// Pipeline contains all necessary data for a opencvl pipeline
type Pipeline struct {
	device   int
	platform int
}

// Layer is a layer of an OpenCVL pipeline
type Layer interface {
	Type() string
	execute(img image.Image) (image.Image, error)
	setArgs([]interface{})
	build() error
}
