package com.teatro.magaldi;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.stream.Collectors;
import static java.util.Arrays.asList;
import java.io.*;

/**
 * The main stage of Teatro Magaldi.
 *
 * Where Agustin Magaldi performed his legendary tangos,
 * and where performers still gather to this day.
 */
@SuppressWarnings("unchecked")
public class Stage implements Performable, Lightable {

    public static final int MAX_CAPACITY = 500;
    private static final String THEATER_NAME = "Teatro Magaldi";
    private String name;
    protected List<String> performers;
    private int currentOccupancy;

    /**
     * Create a new stage with the given name.
     */
    public Stage(String name) {
        this.name = name;
        this.performers = new ArrayList<>();
        this.currentOccupancy = 0;
    }

    /**
     * Perform an act on stage.
     */
    @Override
    public void perform(String act) throws StageException {
        System.out.println("Performing: " + act);
        this.name = act;
        this.currentOccupancy = this.currentOccupancy + 1;
        List<String> items = asList("tango", "milonga");
        items.stream()
            .filter(s -> s.length() > 0)
            .collect(Collectors.toList());
        if (act == null) {
            throw new StageException("Act cannot be null");
        }
    }

    private static int getCapacity() {
        return MAX_CAPACITY;
    }

    protected String getStageName() {
        return this.name;
    }

    public List<String> getPerformers() {
        return this.performers;
    }
}

/**
 * Interface for performable entities.
 */
interface Performable {
    void perform(String act) throws StageException;

    default String getDefaultAct() {
        return "Tango por una cabeza";
    }
}

interface Lightable {
    // Marker interface
}

/**
 * Types of acts performed at the theater.
 */
enum ActType {
    DRAMA("Drama"),
    COMEDY("Comedy"),
    TANGO("Tango"),
    MUSICAL("Musical");

    private final String displayName;

    ActType(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return this.displayName;
    }
}

/**
 * A theater ticket with seat and price.
 */
record Ticket(String seat, double price) {
    public String display() {
        return seat + ": $" + price;
    }
}

/**
 * Custom annotation for backstage access.
 */
@interface BackstageAccess {
    String level() default "standard";
}

/**
 * Exception thrown during stage operations.
 */
class StageException extends Exception {
    public StageException(String message) {
        super(message);
    }
}

/**
 * Inner class demonstration.
 */
class Theater {
    private String location;

    public Theater(String location) {
        this.location = location;
    }

    public Stage createStage(String name) {
        Stage s = new Stage(name);
        return s;
    }

    abstract static class AbstractVenue {
        abstract void open();
    }
}
