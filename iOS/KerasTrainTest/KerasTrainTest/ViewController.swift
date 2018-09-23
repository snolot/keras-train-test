//
//  ViewController.swift
//  KerasTrainTest
//
//  Created by Sylvain Nolot on 23/09/2018.
//  Copyright © 2018 Sylvain Nolot. All rights reserved.
//
import UIKit
import CoreML
class ViewController: UIViewController {
    
    let mapping = [
        "terrible": 1.0,
        "great": 2.0,
        "bad": 3.0,
        "good": 4.0,
        "awful": 5.0,
        "awesome": 6.0
    ]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let text = ["That team is terrible","That play was great","I'm really feeling awful right now"]
        var wordArrays : [[String]] = []
        for sent in text {
            wordArrays.append(sent.characters.split{$0 == " "}.map(String.init))
        }
        
        let model = sentiment_model()
        guard let input_data = try? MLMultiArray(shape:[80,1,1], dataType:.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        guard let gru_1_h_in = try? MLMultiArray(shape:[128], dataType:.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        gru_1_h_in[0] = NSNumber(value: 0.0)
        var list : [[NSNumber]] = []
        for input in wordArrays {
            list.append(tokenizer(words: input))
        }
        
        for (index,item) in list[0].enumerated() {
            input_data[index] = item
        }
        
        let i = sentiment_modelInput(tokenizedString: input_data,gru_1_h_in:gru_1_h_in)
        do {
            let sentiment_prediction = try model.prediction(input: i)
            
            
            if(sentiment_prediction.sentiment[0].doubleValue >= 0.5) {
                print("positive!")
            } else {
                print("negative!")
            }
            
            for index in 1 ... list.count - 1 {
                for (index,item) in list[index].enumerated() {
                    input_data[index] = item
                }
                let next_input = sentiment_modelInput(tokenizedString: input_data,gru_1_h_in:sentiment_prediction.gru_1_h_out)
                let next_prediction = try model.prediction(input: next_input)
                //print(next_prediction.sentiment[0])
                
                if(next_prediction.sentiment[0].doubleValue >= 0.5) {
                    print("positive!")
                } else {
                    print("negative!")
                }
            }
            
        } catch {
            print(error.localizedDescription)
        }
        
    }
    
    func padArray(to numToPad: Int, sequence: [NSNumber]) -> [NSNumber] {
        var newSeq = sequence
        for _ in sequence.count ... numToPad {
            newSeq.insert(NSNumber(value:0.0), at: 0)
        }
        return newSeq
    }
    
    func tokenizer(words: [String]) -> [NSNumber] {
        var tokens : [NSNumber] = []
        for (index, word) in words.enumerated() {
            if let val = mapping[word] {
                tokens.insert(NSNumber(value: val), at: index)
            } else {
                tokens.insert(NSNumber(value: 0.0), at: index)
            }
        }
        return padArray(to: 79, sequence: tokens)
    }
}
