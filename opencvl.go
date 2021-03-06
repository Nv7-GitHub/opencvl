package opencvl

import (
	"errors"
	"image"

	"gocv.io/x/gocv"
)

// Pipeline contains all necessary data for a opencvl pipeline
type Pipeline struct {
	layers []Layer
}

// Layer is a layer of an OpenCVL pipeline
type Layer interface {
	Type() string
	execute(img *image.RGBA, time int) (*image.RGBA, error)
	setArgs([]interface{}) error
	build() error
	cleanup()
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
		return errors.New("opencvl: invalid pipeline kind")
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
func (p *Pipeline) ExecuteOnImage(img *image.RGBA, timeDat ...int) (*image.RGBA, error) {
	time := 0
	if len(timeDat) > 0 {
		time = timeDat[0]
	}
	var err error
	for _, layer := range p.layers {
		img, err = layer.execute(img, time)
		if err != nil {
			return nil, err
		}
	}
	return img, nil
}

// ExecuteOnVideo executes the pipeline on every frame of a video and encodes it into a new video
func (p *Pipeline) ExecuteOnVideo(file string, outFile string) error {
	video, err := gocv.VideoCaptureFile(file)
	if err != nil {
		return err
	}
	defer video.Close()
	img := gocv.NewMat()
	var im image.Image
	video.Read(&img)
	writer, err := gocv.VideoWriterFile(outFile, video.CodecString(), video.Get(gocv.VideoCaptureFPS), img.Cols(), img.Rows(), true)
	if err != nil {
		return err
	}
	defer writer.Close()
	time := 0
	for video.Read(&img) {
		if img.Empty() {
			continue
		}
		im, err = img.ToImage()
		if err != nil {
			return err
		}

		im, err = p.ExecuteOnImage(im.(*image.RGBA))
		if err != nil {
			return err
		}

		img, err = gocv.ImageToMatRGB(im)
		if err != nil {
			return err
		}

		err = writer.Write(img)
		if err != nil {
			return err
		}

		time++
	}
	return nil
}

// AddPipeline allows you to add a pipeline inside of another pipeline
func (p *Pipeline) AddPipeline(pipeline Pipeline) error {
	var err error
	for _, layer := range pipeline.layers {
		err = p.AddLayer(layer)
		if err != nil {
			return err
		}
	}
	return nil
}

// Cleanup cleans up the arguments (only applies to OpenCL layers)
func (p *Pipeline) Cleanup() {
	for _, layer := range p.layers {
		layer.cleanup()
	}
}
