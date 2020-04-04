#pragma once
#ifndef __KDNTREE_H__
#define __KDNTREE_H__

#include <valarray>

typedef std::valarray<double> Point;

class KDNtree
{
private:
	std::uint32_t leaf_size;
	std::valarray<Point> X;
	std::valarray<int> Xi;

	class Leaf
	{

	};

	class Node
	{

	};

public:
	KDNtree();

	void query()

};

#endif //__KDNTREE_H__