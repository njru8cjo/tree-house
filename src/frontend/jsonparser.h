#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include "json.hpp"
#include "decisiontree.h"
#include <fstream>
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using json = nlohmann::json;
using namespace mlir;

namespace Treehierarchy
{
    class BuildOptions
    {
    public:
        bool enable_flint = false;
        bool enable_ra = false;
    };

    class JsonParser
    {
    public:
        JsonParser(const std::string &forestJSONPath, BuildOptions option) : m_option(option), m_forest(new DecisionForest()),
                                                                             m_context(), m_builder(&m_context),
                                                                             m_module(ModuleOp::create(m_builder.getUnknownLoc(), llvm::StringRef("ForestModule")))
        {
            std::ifstream fin(forestJSONPath);
            assert(fin);
            fin >> m_json;
            initMLIRContext(m_context);
        }

        // Provide a virtual destructor definition
        virtual ~JsonParser()
        {
            delete m_forest;
        }

        virtual void ConstructForest() = 0;

        ModuleOp buildHIRModule()
        {

            for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
            {
                OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

                func::FuncOp function(getFunctionPrototype("tree_" + std::to_string(i)));

                Block *entryBlock = function.addEntryBlock();
                DecisionTree *tree = m_forest->GetTree(i);
                m_builder.setInsertionPointToStart(entryBlock);
                buildNodeOp(entryBlock, tree, 0);
                m_module.push_back(function);

                m_builder.restoreInsertionPoint(insertPoint);
            }

            CreatePredictFunction();

            return m_module;
        }

        ModuleOp lowerToLLVMModule()
        {

            LLVMTypeConverter converter(&m_context);

            ConversionTarget target(m_context);
            RewritePatternSet patterns(&m_context);

            target.addLegalDialect<LLVM::LLVMDialect>();
            target.addIllegalDialect<arith::ArithDialect, func::FuncDialect>();

            arith::populateArithToLLVMConversionPatterns(converter, patterns);
            populateFuncToLLVMConversionPatterns(converter, patterns);
            populateMathToLLVMConversionPatterns(converter, patterns);

            if (failed(applyPartialConversion(m_module, target, std::move(patterns))))
            {
                llvm::errs() << "Decision forest lowering pass failed\n";
            }

            return m_module;
        }

    protected:
        BuildOptions m_option;
        json m_json;
        DecisionForest *m_forest;
        DecisionTree *m_decisionTree;
        MLIRContext m_context;
        OpBuilder m_builder;
        ModuleOp m_module;

        virtual void CreatePredictFunction() = 0;

        void initMLIRContext(MLIRContext &context)
        {
            context.getOrLoadDialect<arith::ArithDialect>();
            context.getOrLoadDialect<func::FuncDialect>();
            context.getOrLoadDialect<LLVM::LLVMDialect>();
            context.getOrLoadDialect<math::MathDialect>();
        }

        Type getFeaturePointerType()
        {
            LLVMTypeConverter converter(&m_context);
            if (m_option.enable_flint)
            {
                return converter.getPointerType(getI32());
            }
            else
            {

                return converter.getPointerType(m_builder.getF32Type());
            }
        }

        Type getArrayType()
        {
            return LLVM::LLVMArrayType::get(getF32(), m_forest->GetFeatureSize());
        }

        Type getFeatureType()
        {
            return m_option.enable_flint ? getI32() : getF32();
        }

        Type getI32()
        {
            return m_builder.getI32Type();
        }

        Type getF32()
        {
            return m_builder.getF32Type();
        }

        func::FuncOp getFunctionPrototype(std::string funName)
        {
            auto loc = m_builder.getUnknownLoc();
            Type argType = getFeaturePointerType();
            auto functionType = m_builder.getFunctionType({argType}, getF32());
            auto function = m_builder.create<func::FuncOp>(loc, funName, functionType);
            function.setPublic();

            return function;
        }

        Value createThreshold(float thresholdVal)
        {
            auto loc = m_builder.getUnknownLoc();
            if (m_option.enable_flint)
            {
                if (thresholdVal < 0)
                {
                    thresholdVal *= -1;
                }
                int intValue = *(int *)&thresholdVal;
                return m_builder.create<arith::ConstantIntOp>(loc, intValue, getI32());
            }
            else
            {
                return m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(thresholdVal));
            }
        }

        void buildNodeOp(Block *entryBlock, DecisionTree *tree, int idx)
        {
            DecisionTree::Node node = tree->GetNode(idx);
            auto loc = m_builder.getUnknownLoc();

            if (!node.IsLeaf())
            {
                Value threshold = createThreshold(node.threshold);

                Value featureIdx = m_builder.create<arith::ConstantIntOp>(loc, node.featureIndex, getI32());

                Value input = entryBlock->getArgument(0);

                Value featurePtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getFeatureType(), input, featureIdx);
                Value feature = m_builder.create<LLVM::LoadOp>(loc, getFeatureType(), featurePtr);
                if (m_option.enable_flint && node.threshold < 0)
                {
                    Value mask = m_builder.create<arith::ConstantIntOp>(loc, 0x1 << 31, getI32());
                    feature = m_builder.create<mlir::arith::XOrIOp>(loc, feature, mask);
                }

                OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

                auto leftNode = tree->GetNode(node.leftChild);
                auto rightNode = tree->GetNode(node.rightChild);
                auto predicate = arith::CmpFPredicate::OLT;
                auto predicate2 = LLVM::ICmpPredicate::slt;
                int64_t leftIdx = node.leftChild;
                int64_t rightIdx = node.rightChild;

                if (leftNode.probability < rightNode.probability)
                {
                    predicate = arith::CmpFPredicate::OGE;
                    predicate2 = LLVM::ICmpPredicate::sge;
                    leftIdx = node.rightChild;
                    rightIdx = node.leftChild;
                }

                Value condition;
                if (m_option.enable_flint && node.threshold < 0)
                {
                    condition = m_builder.create<LLVM::ICmpOp>(loc, predicate2, threshold, feature);
                }
                else if (m_option.enable_flint)
                {
                    condition = m_builder.create<LLVM::ICmpOp>(loc, predicate2, feature, threshold);
                }
                else
                {
                    condition = m_builder.create<arith::CmpFOp>(loc, predicate, feature, threshold);
                }

                Region *funcBody = entryBlock->getParent();
                Block *tBlock = m_builder.createBlock(funcBody);
                m_builder.setInsertionPointToStart(tBlock);
                buildNodeOp(entryBlock, tree, leftIdx);

                Block *fBlock = m_builder.createBlock(funcBody);
                m_builder.setInsertionPointToStart(fBlock);
                buildNodeOp(entryBlock, tree, rightIdx);

                m_builder.restoreInsertionPoint(insertPoint);
                ValueRange nullList = {};
                m_builder.create<LLVM::CondBrOp>(loc, condition, tBlock, nullList, fBlock, nullList);
            }
            else
            {
                Value result = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(node.threshold));
                m_builder.create<func::ReturnOp>(loc, result);
            }
        }
    };
}
#endif