const { default: mongoose } = require('mongoose');


 const productSchma =new mongoose.Schema({ // first create schema
    name:String,
    price:String,
    description:String,
    ratting:String,
    images:[
        {
            image:String
        }
    ],
    category:String,
    seller:String,
    stock:String,
    numOfReviews:String,
    createdAt:Date
});

const productModel =mongoose.model('Products',productSchma)//second auto create model when mongodb

module.exports = productModel; // export model