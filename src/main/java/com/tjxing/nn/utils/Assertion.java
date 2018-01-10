package com.tjxing.nn.utils;

public class Assertion {

    public static void assertEqual(int x, int y) {
        if(x != y) {
            throw new AssertionException(Integer.toString(x), "==", Integer.toString(y));
        }
    }

    public static class AssertionException extends RuntimeException {

        public AssertionException(String...args) {
            super("Assertion Failed: " + makeString(args));
        }

        private static String makeString(String...args) {
            StringBuilder sb = new StringBuilder();
            for(String s : args) {
                sb.append(s);
            }
            return sb.toString();
        }

    }

}
