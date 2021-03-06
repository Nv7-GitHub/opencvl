

__kernel void mandelbrot(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int time,
		const int animLength)
{
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	const int2 dim = get_image_dim(in);
	const float2 pf = convert_float2(pos)/convert_float2(dim);
	if (time == 0) {
		pf.x = pf.x*4-2.5;
		pf.y = pf.y*2.5-1.25;
	} else {
		float multiplier = pow(convert_float(animLength-(time+1))/convert_float(animLength), 2);
		pf.x = pf.x*(4*multiplier)-(2.5*(multiplier/2+0.5));
		pf.y = pf.y*(2.5*multiplier)-(1.25*(multiplier/1.1 + (1 - 1/1.1)));
		if (pos.x == 0 && pos.y == 0) {
			printf("%d\n", time);
		}
	}
	float4 pixel = (float4)(0);
	int i;
	float2 z = (float2)(0);
	float color = 1;
	float xtemp;
	//const int maxLoops = time+50;
	float maxLoops = 100;
	if (pos.x < dim.x && pos.y < dim.y) {
		for (i = 0; i < maxLoops; i++) {
			xtemp = (z.x * z.x) - (z.y*z.y);
			z.y = (2 * z.x * z.y);
			z.x = xtemp;
			z += pf;
			if (length(z) > 2.0) {
				color = i/maxLoops;
				break;
			}
		}
		pixel = (float4)((float3)(1-color), (float)1);
		if (color == 1) {
			pixel.z = 0;
		}
		write_imagef(out, pos, pixel);
	}
}

