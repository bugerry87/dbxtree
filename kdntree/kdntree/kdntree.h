#pragma once
#ifndef __KDNTREE_H__
#define __KDNTREE_H__

#include <valarray>
#include <stack>
#include <thread>

#if _WIN64 || __x86_64__ || __ppc64__
#define __ENV64__
typedef std::uint64_t uint;
typedef std::double_t real;
#else
#define __ENV32__
typedef std::uint32_t uint;
typedef std::float_t real;
#endif


typedef std::valarray<uint> Index;
typedef std::valarray<Index> IndexArr;
typedef std::valarray<real> Point;
typedef std::valarray<Point> PointArr;

class KDNtree
{
protected:
	struct Node
	{
		virtual void query(KDNtree& tree, const PointArr P) = 0;
	};

	struct Branch : public Node
	{
		IndexArr Xi;
		Node* left;
		Node* center;
		Node* right;

		Branch();

		void query(KDNtree& tree, const PointArr P) override;

	private:
		void expand(KDNtree& tree);
	};

	struct Leaf : public Node
	{
		uint leaf_size;
		IndexArr Xi;

		void query(KDNtree& tree, const PointArr P) override;
	};

	typedef std::stack<Node*> Stack;

	PointArr X;
	Stack node_stack;
	Node* root;

public:
	uint leaf_size;
	Point L;
	PointArr Y;
	IndexArr nn;

	KDNtree(uint leaf_size = 0);

	void fit(const PointArr& X, const IndexArr& Xi, uint leaf_size = 0);

	static void fit(KDNtree& tree, const PointArr& X, const IndexArr& Xi, uint leaf_size = 0);

	void query(const PointArr& P, const uint j = 0);

	static void query(KDNtree& tree, const PointArr& P, const uint j = 0);
};

#endif //__KDNTREE_H__