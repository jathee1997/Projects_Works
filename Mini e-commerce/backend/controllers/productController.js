const productModel = require("../models/productModel")

//Get Products API - http://api/v1/products
exports.getProducts=async(req,res,next)=>{ //if  we used await key we must get async

   const query=req.query.keyword?{ name: {

      $regex: req.query.keyword,// regex its mean  reqularoperation  operater
      $options: 'i'
   }}:{}

   const products = await productModel.find(query);//get data from Productmodel(db ) ///and assigingranate method
   res.json({

    success:true,
    products
     //message:'Get products working' //{before set model}(not need that function const products = await productModel.find({}))
 })


}

//Get Products API - http://api/v1/product/:id
exports.getSingleProducts=async(req,res,next)=>{
   try{
      const product = await productModel.findById(req.params.id);//get product data via variable product
      res.json({
     
         success:true,
         product
        // message:'Get single products working'//{before set model}(not need that function const products = await productModel.find({}))
      })
      console.log(req.params.id,'ID') //test get id from MDB
   }catch(error){

      res.status(404).json({
     
         success:false,
         message:error.message,//or
         message:'unable to get product with that id'
        // message:'Get single products working'//{before set model}(not need that function const products = await productModel.find({}))
      })

      
   }

  
   
   
   }