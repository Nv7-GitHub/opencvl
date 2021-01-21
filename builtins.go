package opencvl

// Builtins contains some simple pipelines

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

// BlurPipeline performs a gaussian blur, given the magnitude of the blur on the x and y axis
func BlurPipeline(xblur, yblur int) Pipeline {
	p := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.GaussianBlur(mat, &mat, image.Point{xblur, yblur}, 0, 0, gocv.BorderDefault)
		return mat
	})
	p.AddLayer(layer)
	return p
}

// InvertPipeline inverts an image
func InvertPipeline() Pipeline {
	p := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.BitwiseNot(mat, &mat)
		return mat
	})
	p.AddLayer(layer)
	return p
}

// HSLCorrectPipeline does an HSL color correction pipeline
func HSLCorrectPipeline(hchange, schange, lchange float64) Pipeline {
	p := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.CvtColor(mat, &mat, gocv.ColorBGRToHLS)
		mask := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(hchange, lchange, schange, 0), mat.Rows(), mat.Cols(), mat.Type())
		gocv.Add(mat, mask, &mat)
		gocv.CvtColor(mat, &mat, gocv.ColorHLSToBGR)
		return mat
	})
	p.AddLayer(layer)
	return p
}

// TranslatePipeline translates an image
func TranslatePipeline(xchange, ychange int) (Pipeline, error) {
	p := NewPipeline()
	var kernelSource = fmt.Sprintf(`
	__kernel void translate(
		__read_only image2d_t in,
		__write_only image2d_t out)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		const int2 dim = get_image_dim(in);
		float4 pixel = (float4)(0);
		if (pos.x < dim.x && pos.y < dim.y) {
			pixel = read_imagef(in, pos);
			pos.x += %d;
			pos.y += %d;
			if (pos.x < dim.x && pos.y < dim.y) {
				write_imagef(out, pos, pixel);
			}
		}
	}`, xchange, ychange)
	layer, err := NewOpenCLLayer(kernelSource, "translate", 0, 0)
	if err != nil {
		return NewPipeline(), err
	}
	p.AddLayer(layer)
	err = p.Build()
	if err != nil {
		return NewPipeline(), err
	}
	return p, nil
}

// RotatePipeline rotates an image
func RotatePipeline(x, y int, rotation float64) Pipeline {
	p := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		matrix := gocv.GetRotationMatrix2D(image.Point{x, y}, rotation, 1)
		gocv.WarpAffine(mat, &mat, matrix, image.Point{0, 0})
		return mat
	})
	p.AddLayer(layer)
	return p
}
