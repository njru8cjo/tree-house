

#include "decisiontree.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace mlir
{
    namespace decisionTree
    {
        class ModuleBuilder
        {

        public:
            ModuleBuilder(DecisionForest *decisionForest)
                : m_forest(decisionForest),
                  m_context(),
                  m_builder(&m_context),
                  m_module(ModuleOp::create(m_builder.getUnknownLoc(), llvm::StringRef("ForestModule")))
            {
                initMLIRContext(m_context);
            }

            ModuleOp buildHIRModule()
            {

                for (size_t i = 0; i < m_forest->getTreeSize(); i++)
                {

                    OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

                    func::FuncOp function(getFunctionPrototype("tree_" + std::to_string(i)));

                    Block *entryBlock = function.addEntryBlock();
                    DecisionTree *tree = m_forest->getTree(i);
                    m_builder.setInsertionPointToStart(entryBlock);
                    buildNodeOp(entryBlock, tree, 0);
                    m_module.push_back(function);

                    m_builder.restoreInsertionPoint(insertPoint);
                }

                func::FuncOp mainFun(getFunctionPrototype("predict"));
                Block *callerBlock = mainFun.addEntryBlock();
                m_builder.setInsertionPointToStart(callerBlock);

                Location loc = m_builder.getUnknownLoc();
                Value sum = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(0.0));

                for (size_t i = 0; i < m_forest->getTreeSize(); i++)
                {
                    auto callResult = m_builder.create<func::CallOp>(loc, StringRef("tree_" + std::to_string(i)), getF32(), callerBlock->getArgument(0));
                    sum = m_builder.create<arith::AddFOp>(loc, sum, callResult.getResult(0));
                }

                m_builder.create<func::ReturnOp>(loc, sum);
                m_module.push_back(mainFun);

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

                if (failed(applyPartialConversion(m_module, target, std::move(patterns))))
                {
                    llvm::errs() << "Decision forest lowering pass failed\n";
                }

                return m_module;
            }

        private:
            DecisionForest *m_forest;
            MLIRContext m_context;
            OpBuilder m_builder;
            ModuleOp m_module;

            void initMLIRContext(MLIRContext &context)
            {
                context.getOrLoadDialect<arith::ArithDialect>();
                context.getOrLoadDialect<func::FuncDialect>();
                context.getOrLoadDialect<LLVM::LLVMDialect>();
            }

            Type getFunctionArgumentType()
            {
                LLVMTypeConverter converter(&m_context);
                return converter.getPointerType(m_builder.getF32Type());
            }

            Type getArrayType()
            {
                return LLVM::LLVMArrayType::get(m_builder.getF32Type(), m_forest->getFeatureSize());
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
                Type argType = getFunctionArgumentType();
                auto functionType = m_builder.getFunctionType({argType}, getF32());
                auto function = m_builder.create<func::FuncOp>(loc, funName, functionType);
                function.setPublic();

                return function;
            }

            void buildNodeOp(Block *entryBlock, DecisionTree *tree, int idx)
            {
                DecisionTree::Node node = tree->getNode(idx);
                auto loc = m_builder.getUnknownLoc();

                Value threshold = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(node.threshold));
                auto op = threshold.getDefiningOp();
                op->setAttr("prob", FloatAttr::get(getF32(), node.probability));

                if (!node.isLeaf())
                {

                    Value input = entryBlock->getArgument(0);
                    Region *funcBody = entryBlock->getParent();

                    Value featureIdx = m_builder.create<arith::ConstantIntOp>(loc, node.featureIndex, getI32());
                    Value ptr = m_builder.create<LLVM::GEPOp>(loc, getFunctionArgumentType(), getArrayType(), input, featureIdx);
                    Value feature = m_builder.create<LLVM::LoadOp>(loc, getF32(), ptr);
                    OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

                    auto leftNode = tree->getNode(node.leftChild);
                    auto rightNode = tree->getNode(node.rightChild);
                    auto predicate = arith::CmpFPredicate::OLT;
                    int64_t leftIdx = node.leftChild;
                    int64_t rightIdx = node.rightChild;

                    if (leftNode.probability < rightNode.probability)
                    {
                        predicate = arith::CmpFPredicate::OGE;
                        leftIdx = node.rightChild;
                        rightIdx = node.leftChild;
                    }

                    Value condition = m_builder.create<arith::CmpFOp>(loc, predicate, feature, threshold);

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
                    m_builder.create<func::ReturnOp>(loc, threshold);
                }
            }
        };

    }
}