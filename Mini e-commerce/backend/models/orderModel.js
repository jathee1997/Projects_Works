

const  mongoose = require('mongoose');


 const orderSchema =new mongoose.Schema({ // second create schema
   cartItems: Array,
   amount:String,
   status: String,
   createAt:Date
});

const orderModel =mongoose.model('Order',orderSchema)//second auto create model when mongodb

module.exports = orderModel; // export model