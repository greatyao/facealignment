#ifndef ESRBBOX
#define ESRBBOX

namespace ESR
{
	/**
	 * Bounding Box defined in image space
	 */
	struct Bbox
	{
		double sx; //start x
		double sy; //start y
		double cx; //center x
		double cy; //center y
		double w;  //width
		double h;  //height
		Bbox(): sx(0),sy(0),cx(0),cy(0),w(0),h(0){}
		Bbox(double sx, double sy, double cx, double cy, double w, double h): 
		sx(sx),sy(sy),cx(cx),cy(cy),w(w),h(h){}

		void scale(float factor)
		{
			w *= factor;
			h *= factor;
			sx = cx - w/2.0;
			sy = cy - h/2.0;
			return;
		}
		
		void translate(double x, double y)
		{
			sx += x; 
			cx += x; 
			sy += y; 
			cy += y;
			return;
		}

	};
}

#endif