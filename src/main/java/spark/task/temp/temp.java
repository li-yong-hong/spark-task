package spark.task.temp;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by yonghongli on 2016/8/15.
 */
public class temp {
    public static void main(String[] args){
        HashSet<String> hs = new HashSet<String>();
        hs.add("fa");
        hs.add("fa");
        hs.add("sa");
        String[] levelString = new String[hs.size()];
        hs.toArray(levelString);
        System.out.println(levelString.length+levelString[0] +levelString[1] );
    }
}
