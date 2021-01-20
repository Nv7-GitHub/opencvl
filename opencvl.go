package opencvl

import (
	"errors"
	"image"
)

// Pipeline contains all necessary data for a opencvl pipeline
type Pipeline struct {
	layers []Layer
}

// Layer is a layer of an OpenCVL pipeline
type Layer interface {
	Type() string
	execute(img *image.RGBA) (*image.RGBA, error)
	setArgs([]interface{}) error
	build() error
}

// NewPipeline creates a new Pipeline
func NewPipeline() Pipeline {
	return Pipeline{
		layers: make([]Layer, 0),
	}
}

// AddLayer adds a layer to a pipeline
func (p *Pipeline) AddLayer(layer Layer) error {
	kind := layer.Type()
	if kind != "opencv" && kind != "opencl" {
		return errors.New("Invalid pipeline kind")
	}
	p.layers = append(p.layers, layer)
	return nil
}

// SetOpenCVArgs sets the arguments for all OpenCV layers
func (p *Pipeline) SetOpenCVArgs(args ...interface{}) {
	for _, layer := range p.layers {
		if layer.Type() == "opencv" {
			layer.setArgs(args)
		}
	}
}

// SetOpenCLArgs sets the arguments for all OpenCL layers
func (p *Pipeline) SetOpenCLArgs(args ...interface{}) error {
	var err error
	for _, layer := range p.layers {
		if layer.Type() == "opencl" {
			err = layer.setArgs(args)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// Build compiles the pipeline
func (p *Pipeline) Build() error {
	var err error
	for _, layer := range p.layers {
		err = layer.build()
		if err != nil {
			return err
		}
	}
	return nil
}

// ExecuteOnImage executes the pipeline on an image
func (p *Pipeline) ExecuteOnImage(img *image.RGBA) (*image.RGBA, error) {
	var err error
	for _, layer := range p.layers {
		img, err = layer.execute(img)
		if err != nil {
			return nil, err
		}
	}
	return img, nil
}
